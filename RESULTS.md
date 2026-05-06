# CIFAR-100 Speedrun 5m Results

Current best: run `xp-bd9ca3` reached `val_accuracy=0.670600` with `val_loss=1.246395`, `n_params=3,423,620`, and `train_seconds=270.853`.

I first attempted the same ResNet-56 width-2 recipe in `xp-bc9442`, but it crashed before training because the mounted parquet path produced pyarrow thrift deserialization errors. I fixed data loading to search mounted train shards recursively and fall back to the Hub train parquet URL without reading the reserved test split.

The first usable model is a widened CIFAR ResNet-56 with SGD/Nesterov, OneCycleLR at `max_lr=0.42`, RandAugment, random erasing, mixup alpha `0.8`, label smoothing `0.08`, BF16 autocast, and EMA. It completed 41 epochs under the 5-minute job cap and landed at 67.06% validation accuracy, above the preregistered 58.0% prediction.

Key finding: this short-budget regime is already near the plausible 65-70% ceiling with a plain CNN and careful recipe. The train accuracy proxy is not directly interpretable because batches use mixup, but EMA validation accuracy is strong.

Next hypothesis: keep the CNN backbone and tune recipe overhead. Candidate pushes are reducing CPU-heavy RandAugment, trying batch size 768 or 1024, adjusting OneCycle `max_lr`, or replacing mixup-only with CutMix to improve late validation accuracy without slowing the loader.
