---
name: ot-vqa-diagnostics
description: Diagnose OT-VQA training runs, checkpoint behavior, modality reliance, and Sinkhorn transport health using the repository's diagnostic CLI. Use when training is unstable or quality regresses.
---

# OT-VQA diagnostics

Read `diagnose_training.py --help` and the checkpoint's `run_config.json`/`metrics.jsonl` first. The diagnostic CLI needs explicit validation CSV and image paths; it does not accept `--dataset` or `--split`.

```bash
python diagnose_training.py \
  --checkpoint data/scivqa_model/best.pt \
  --dev_csv_path data/scivqa_dataset/val.csv \
  --dev_img_path data/scivqa_dataset/images \
  --samples 32 --device auto
```

Use `--train_csv_path` when the requested diagnostic compares training and validation behavior. Use `--metrics` only when the metrics file is outside the checkpoint directory; otherwise the script looks beside the checkpoint. `--output` writes a JSON report.

Inspect validation trends, output diversity, modality reliance, valid-token masks, transport mass, and row/column residuals as applicable. Follow evidence to a likely cause before changing hyperparameters. A high OT score or visually plausible plan alone does not prove that answers use image evidence; compare controlled validation behavior.
