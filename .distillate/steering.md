# Steering

predicted_metric MUST be the brief's primary metric (declared under **Optimize:** in PROMPT.md, also in .distillate/experiment.json::metric_name). Do NOT predict on the gate metric. For tinymatmul3x3 use predicted_metric=n_params; for cifar-100 use predicted_metric=test_accuracy. Re-read CLAUDE.md / AGENTS.md / GEMINI.md from .distillate/ — the prereg-discipline section was just updated. This applies to every prereg from now on.
