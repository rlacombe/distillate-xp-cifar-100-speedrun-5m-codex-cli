# Task: CIFAR-100 Speedrun 5m (Codex CLI)

**Objective:** What is the highest CIFAR-100 validation accuracy achievable by an image classifier with ≤5M parameters when training is wall-clock-capped at 5 minutes on a single H200?

## Research Question

What is the highest CIFAR-100 validation accuracy achievable by an image classifier with ≤5M parameters when training is wall-clock-capped at 5 minutes on a single H200?

## Context

CIFAR-100 image classification — 100 fine-grained classes, 32×32 RGB, 50K train / 10K test. Standard torchvision split: 45K train / 5K val carved from the 50K training set with a fixed seed; the 10K original test set is RESERVED for paper conclusion only and must never be read or scored against during the campaign. Vocabulary: 100 class indices.

Architecture surface is open: CIFAR-style ResNet-20/32/56, WideResNet, ViT-Tiny (patch-2 / patch-4 over 32×32), MobileNet, ConvMixer, MLP-Mixer, hybrid CNN/attention — any classifier built from learned weights via gradient descent. Training recipe is part of the research surface and matters as much as architecture: optimizer (SGD-momentum, AdamW, Lion, Lookahead), schedule (cosine, OneCycle, warmup), augmentations (random crop, hflip, AutoAugment, RandAugment, Mixup, CutMix, Cutout), regularization (label smoothing, weight decay, dropout, stochastic depth), mixed precision, gradient accumulation, EMA — all explicitly allowed.

This experiment is replicated across three harnesses (Claude Code / Codex CLI / Gemini CLI) for cross-agent calibration; each agent runs ~6 hours / ~30-50 runs producing a bit-identical schema in its own runs.jsonl. The three deployments share a bit-identical PROMPT.md, train.py template, evaluator, budget, and pre-cached dataset; only the agent (model + harness) differs.

Calibration intent — read carefully. This is an accuracy-maximization protocol with FRONTIER-PUSH PRE-REGISTRATION. Pre-register the predicted peak honestly, not the gate. A *frontier-push prediction* is one where `predicted_value` strictly exceeds the running maximum from prior runs — when you make one, you are claiming this run will set a new peak. Use the full confidence range: a bold but uncertain frontier-push (predicted 0.65 / conf 40) is more informative than a timid one (predicted 0.56 / conf 70). Treat 5 min as the actual training budget, not a starting point — architectures that don't converge in this budget are not in the search space.

## Optimization

**Optimize:** `val_accuracy` (maximize)

**Subject to (gates):**
- `n_params` ≤ `5000000`
- `train_seconds` ≤ `300`

**Intent:** Push CIFAR-100 val accuracy upward run-by-run inside a hard 5-min H200 / ≤5M-param envelope, pre-registering a numeric `predicted_value` for `val_accuracy` before each run. Cross-vendor calibration analysis (Table 5 of the calibration paper) scores: (a) frontier-push hit rate P(Δy>0 | Δŷ>0), (b) magnitude correlation r(Δŷ, Δy | F), (c) confidence-to-outcome correlation r(c, Δy | F) within the frontier-push set F. Predictions and confidence are tracked for calibration; calibrate honestly — pre-register the value you actually expect, not the value you wish for.

**Baseline:** CIFAR-style ResNet-20 (~270K params) with SGD-momentum + cosine schedule + standard augs (random crop + hflip): ~55-60% val accuracy in 5 min on H200. ResNet-56 / WideResNet-16-8 (~1-3M params) with mixup and stronger augs: ~60-65%. Beyond that, recipe matters as much as architecture — AutoAugment, AdamW, OneCycleLR, label smoothing, EMA all carry weight at this budget. SOTA at full convergence is ~78% for sub-5M-param models; the 5-min ceiling is unknown but plausibly 65-70% with the right recipe.

## Setup

### Training

Hardware: single NVIDIA H200 on Hugging Face Jobs. Each training run is dispatched as one HF Job invocation; train.py is auto-uploaded to HF Hub and run inside a managed container.

Job submission contract — via the Distillate MCP server, NEVER bare `huggingface-cli jobs run`:
1. `start_run(...)` — pre-register prediction + hypothesis (writes prereg row to .distillate/runs.jsonl + commits).
2. `submit_hf_job(project, script="train.py", gpu_flavor="h200", timeout_minutes=5, env={"TRAIN_BUDGET_SECONDS": "300"}, volumes=["hf://datasets/uoft-cs/cifar100:/data"])` — auto-uploads script, mounts dataset at /data, returns job_id immediately (non-blocking).
3. `tail_hf_job_logs(job_id)` or `check_hf_job(job_id)` — train.py must emit final metrics as `METRIC <key>=<value>` stdout lines (e.g. `METRIC val_accuracy=0.612`).
4. `conclude_run(...)` — write run row + commit using `suggested_commit_msg`; push.
5. `cancel_hf_job(job_id)` available for early termination.

Per-run training budget: 5 min wall-clock — train_budget_seconds=300, wrap_budget_seconds=60 in .distillate/budget.json. HF Jobs server-side `timeout_minutes` is the hard SIGTERM gate; an in-script wall-clock guard inside train.py reads `int(os.environ.get("TRAIN_BUDGET_SECONDS", "300")) - 30` (30s reserve for eval/checkpoint) and exits cleanly on epoch boundaries before SIGTERM lands. NEVER hardcode the budget.

BF16 mixed precision on by default (H200 has dedicated BF16 throughput; small-batch CIFAR pipeline benefits directly). PyTorch 2.x with `torch.compile(mode="reduce-overhead")` allowed and encouraged for repeated short runs. Declare deps via `# requirements: torch torchvision …` top-of-file comment — HF Jobs reads this and installs via `uv`.

Data: HF dataset `uoft-cs/cifar100` (https://huggingface.co/datasets/uoft-cs/cifar100), parquet format with columns `img` (PIL.Image, 32×32 RGB), `fine_label` (int 0-99 — use this as the 100-class target), `coarse_label` (int 0-19 — ignore). Splits: `train` (50K) and `test` (10K). The dataset repo is mounted at `/data` by the `volumes=["hf://datasets/uoft-cs/cifar100:/data"]` arg to `submit_hf_job`; the parquet shards land under `/data/cifar100/*.parquet`. Load with `datasets.load_dataset("parquet", data_files={"train": "/data/cifar100/train-*.parquet", "test": "/data/cifar100/test-*.parquet"})` or read shards directly with pyarrow — DO NOT call `datasets.load_dataset("uoft-cs/cifar100")` without `data_files=` pointing at `/data`, that re-downloads. 45K/5K train/val split derived from the 50K `train` split with a fixed seed (declare it in `train.py`). The 10K `test` split is RESERVED for paper conclusion only — never read or score during the campaign. DataLoader: num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2 (otherwise the GPU sits idle waiting on data and the 5-min budget evaporates).

One run config per training-script invocation — no internal hyperparameter sweeps within a single start_run/conclude_run cycle. Each iteration = one architecture + one recipe + one seed. Keep individual Python files under 400 lines (51,200-byte tool read limit).

### Model

Any image classifier built from learned weights via gradient descent — CNN, ViT, hybrid, MLP-mixer, etc. Input 3×32×32, output 100-class logits. HARD CAP: ≤5M parameters.

Architectures that route through pretrained backbones, frozen features, or non-learned components are NOT valid solutions and will be rejected at conclude_run. All weights must be trained from random initialization on CIFAR-100 in this run.

Training recipe is part of the research surface — optimizer, schedule, augmentations, regularization, mixed precision, gradient accumulation, EMA all explicitly allowed and explicitly part of what should be explored.

## Run Budget

Read from `.distillate/budget.json` (system default).
**Harness:** codex

## Workspace API Reference

The following tools live alongside this experiment. Their full contract is below — **do not leave the workspace to read their source**.

### `distillate-run` wrapper

A thin process supervisor on `$PATH`. Invoke training as:

```bash
distillate-run python3 train.py [args...]
```

Reads `.distillate/budget.json::train_budget_seconds`, `exec`s the supplied command, sends `SIGTERM` at the deadline and `SIGKILL` after a short grace window. **Always** launch training through it — never bare `python3 train.py`. The wrapper writes nothing else; your training script stays in charge of stdout/stderr/checkpoints/eval.

### `read_train_budget()` — the in-script timer

```python
from distillate.budget import read_train_budget
MAX_SECONDS = read_train_budget()  # int, seconds
```

Returns `train_budget_seconds − 300` (a 300s reserve for eval/checkpoint/`conclude_run`). Walks up from `cwd` looking for `.distillate/budget.json`. Floors at 60s. Falls back to 3300s if no budget file is reachable. Use it for an in-loop wall-clock guard; never hardcode the value.

### `.distillate/budget.json` — schema

```json
{"train_budget_seconds": 540, "wrap_budget_seconds": 60}
```

Edited via the desktop UI. `train_budget_seconds` is the SIGTERM deadline; `wrap_budget_seconds` is reserved post-training time. Don't write this file from the agent — read only.

### `.distillate/runs.jsonl` — append-only run log

One JSON line per training run. Append after each iteration; never rewrite earlier lines. See **Recording Results** below for the schema.

## Experiment Tracking (Distillate)

### Prior Runs
Before starting, **read `.distillate/runs.jsonl`** if it exists. It contains the history of all prior iterations. Build on what worked, avoid repeating failed approaches. Reference prior run IDs in your reasoning. If `.distillate/context.md` exists, read it for a formatted summary.

### Recording Results
After each iteration, you MUST append one JSON line to `.distillate/runs.jsonl`:

```json
{"$schema":"distillate/run/v1", "id":"run_NNN", "timestamp":"ISO8601", "status":"keep|discard|crash", "hypothesis":"...", "changes":"...", "hyperparameters":{...}, "results":{...}, "reasoning":"..."}
```

Set `status` to `keep` if results improved, `discard` if not, `crash` on failure. Include `reasoning` to explain your decision.

**CRITICAL: File Size Limit.** Tool reads must not exceed 51,200 bytes. For files longer than ~400 lines, always use `offset`/`limit`. Keep individual Python files under 400 lines.

## Primary Metric

Metric: val_accuracy (maximize)

Distillate ranks runs by this metric. Test-set scores are reserved for experiment conclusion; gates above must be on val.
