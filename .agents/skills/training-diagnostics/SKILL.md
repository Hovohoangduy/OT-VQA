---
name: training-diagnostics
description: Diagnose training stability, validation regressions, modality reliance, and OT transport behavior in this repository. Use when a run collapses, diverges, overfits, or underuses images.
---

# OT-VQA training diagnostics

Treat output collapse, overfitting, and weak visual grounding as hypotheses to test, not assumed causes. Falling cross-entropy does not identify a failure mode by itself. Inspect generated answers, validation history, checkpoint configuration, and transport diagnostics together.

## Repository diagnostic command

`diagnose_training.py` requires explicit validation paths. It does not accept `--dataset` or `--split`:

```bash
python diagnose_training.py \
  --checkpoint data/scivqa_model/best.pt \
  --dev_csv_path data/scivqa_dataset/val.csv \
  --dev_img_path data/scivqa_dataset/images \
  --samples 32 --device auto
```

Use `--train_csv_path` when the analysis needs training data, `--metrics` for a non-default history file, and `--output` to save the report. Confirm the available options with `python diagnose_training.py --help`.

## Evidence to inspect

- Train and validation curves, selected epoch, and selection metric.
- Examples of generated answers, empty/EOS outputs, repetition, and answer diversity.
- Performance under controlled image replacement or masking when supported by the diagnostic code.
- OT cost and plan health: finite values, valid-token masking, dustbin mass, and marginal residuals.
- Whether the checkpoint's encoder, image transform, loss normalization, and OT settings match the run being analyzed.

Do not change epsilon, dustbin mass, loss weights, or decoding rules solely from generic heuristics. Form one testable hypothesis, change one factor in a fresh run, and compare on validation. A proposed auxiliary OT objective is an experiment, not a guaranteed cure.
