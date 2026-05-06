import glob
import math
import os
import random
import subprocess
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


def ensure_deps():
    missing = []
    for mod, pkg in [
        ("torchvision", "torchvision==0.21.0"),
        ("datasets", "datasets"),
        ("pyarrow", "pyarrow"),
        ("PIL", "pillow"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


ensure_deps()

from datasets import load_dataset
from PIL import Image
from torchvision import transforms

try:
    from distillate.budget import read_train_budget
except Exception:
    def read_train_budget():
        return int(os.environ.get("TRAIN_BUDGET_SECONDS", "300"))


SEED = 20260506
SPLIT_SEED = 12345
N_CLASSES = 100
MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)


class CifarParquet(Dataset):
    def __init__(self, hf_ds, indices, transform):
        self.ds = Subset(hf_ds, indices)
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item["img"]
        if not isinstance(img, Image.Image):
            img = Image.open(img).convert("RGB")
        return self.transform(img), int(item["fine_label"])


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, drop=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.drop = drop
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = F.dropout(out, p=self.drop, training=self.training)
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return out + self.shortcut(x)


class CifarResNet(nn.Module):
    def __init__(self, depth=56, width=2, drop=0.05):
        super().__init__()
        n = (depth - 2) // 6
        ch = [16 * width, 32 * width, 64 * width]
        self.stem = nn.Conv2d(3, ch[0], 3, 1, 1, bias=False)
        self.in_ch = ch[0]
        self.layer1 = self._stage(ch[0], n, 1, drop)
        self.layer2 = self._stage(ch[1], n, 2, drop)
        self.layer3 = self._stage(ch[2], n, 2, drop)
        self.bn = nn.BatchNorm2d(ch[2])
        self.fc = nn.Linear(ch[2], N_CLASSES)

    def _stage(self, out_ch, blocks, stride, drop):
        layers = [BasicBlock(self.in_ch, out_ch, stride, drop)]
        self.in_ch = out_ch
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_ch, out_ch, 1, drop))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.relu(self.bn(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


class ModelEma:
    def __init__(self, model, decay=0.997):
        import copy
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        ema_state = self.module.state_dict()
        for k, v in ema_state.items():
            raw_k = k.removeprefix("_orig_mod.")
            src = msd.get(raw_k)
            if src is None:
                src = msd.get(f"_orig_mod.{raw_k}")
            if src is None:
                raise KeyError(raw_k)
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(src.detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(src)


def mix_targets(y, num_classes, lam, index, smoothing=0.08):
    off = smoothing / num_classes
    on = 1.0 - smoothing + off
    y1 = torch.full((y.size(0), num_classes), off, device=y.device)
    y2 = torch.full((y.size(0), num_classes), off, device=y.device)
    y1.scatter_(1, y[:, None], on)
    y2.scatter_(1, y[index, None], on)
    return y1.mul(lam).add_(y2, alpha=1.0 - lam)


def soft_ce(logits, target):
    return -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def rand_bbox(size, lam):
    _, _, h, w = size
    cut = math.sqrt(1.0 - lam)
    cut_w, cut_h = int(w * cut), int(h * cut)
    cx, cy = random.randrange(w), random.randrange(h)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, w)
    y2 = min(cy + cut_h // 2, h)
    return x1, y1, x2, y2


def load_train_val():
    files = sorted(glob.glob("/data/cifar100/train-*.parquet"))
    files += sorted(glob.glob("/data/cifar100/train/*.parquet"))
    files += sorted(glob.glob("/data/**/train/*.parquet", recursive=True))
    if not files:
        files = sorted(glob.glob("data/cifar100/train-*.parquet"))
        files += sorted(glob.glob("data/cifar100/train/*.parquet"))
    if not files:
        files = ["https://huggingface.co/datasets/uoft-cs/cifar100/resolve/refs%2Fconvert%2Fparquet/cifar100/train/0000.parquet"]
    try:
        ds = load_dataset("parquet", data_files={"train": files}, split="train")
    except Exception as exc:
        print(f"local_parquet_load_failed={type(exc).__name__}: {exc}", flush=True)
        url = "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/refs%2Fconvert%2Fparquet/cifar100/train/0000.parquet"
        ds = load_dataset("parquet", data_files={"train": url}, split="train")
    g = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(len(ds), generator=g).tolist()
    val_idx, train_idx = perm[:5000], perm[5000:]
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.18), ratio=(0.3, 3.3), value=0),
    ])
    val_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return CifarParquet(ds, train_idx, train_tf), CifarParquet(ds, val_idx, val_tf)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    budget = int(os.environ.get("TRAIN_BUDGET_SECONDS") or read_train_budget())
    train_seconds = max(60, budget - 30)
    start = time.time()

    train_set, val_set = load_train_val()
    workers = int(os.environ.get("NUM_WORKERS", "4"))
    train_loader = DataLoader(
        train_set, batch_size=512, shuffle=True, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0, prefetch_factor=2 if workers > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=1024, shuffle=False, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0, prefetch_factor=2 if workers > 0 else None,
    )

    model = CifarResNet(depth=32, width=3, drop=0.04).to(device)
    model = model.to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    if n_params > 5_000_000:
        raise RuntimeError(f"parameter cap exceeded: {n_params}")
    if device.type == "cuda" and os.environ.get("TORCH_COMPILE", "0") == "1":
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as exc:
            print(f"compile_skipped={exc}")
    ema_source = getattr(model, "_orig_mod", model)
    ema = ModelEma(ema_source, decay=0.997)
    opt = torch.optim.SGD(model.parameters(), lr=0.36, momentum=0.9, weight_decay=5e-5, nesterov=True)
    max_epochs = 220
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=0.36, epochs=max_epochs, steps_per_epoch=len(train_loader),
        pct_start=0.12, div_factor=20.0, final_div_factor=200.0,
    )

    epochs_done = 0
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    for epoch in range(max_epochs):
        model.train()
        loss_sum = seen = correct = 0
        for x, y in train_loader:
            if time.time() - start > train_seconds:
                break
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            idx = torch.randperm(y.size(0), device=device)
            lam = float(torch.distributions.Beta(0.8, 0.8).sample())
            x = x.mul(lam).add_(x[idx], alpha=1.0 - lam)
            target = mix_targets(y, N_CLASSES, lam, idx, smoothing=0.03)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(x)
                loss = soft_ce(logits, target)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.size(0)
        epochs_done = epoch + 1
        if seen:
            print(f"epoch={epochs_done} train_loss={loss_sum/seen:.4f} train_acc_proxy={correct/seen:.4f}", flush=True)
        if time.time() - start > train_seconds:
            break

    val_loss, val_acc = evaluate(ema.module, val_loader, device)
    elapsed = time.time() - start
    print(f"METRIC val_loss={val_loss:.6f}")
    print(f"METRIC val_accuracy={val_acc:.6f}")
    print(f"METRIC n_params={n_params}")
    print(f"METRIC train_seconds={elapsed:.3f}")
    print(f"METRIC epochs={epochs_done}")


if __name__ == "__main__":
    main()
