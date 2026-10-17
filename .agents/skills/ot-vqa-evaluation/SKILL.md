---
name: ot-vqa-evaluation
description: Evaluate OT-VQA checkpoints and compare generated answers using supported dataset splits, report files, and metrics. Use when scoring or comparing a trained model.
---

# OT-VQA evaluation

The `test.py` CLI supports `--split dev` and `--split test`; use `dev` for model selection and iteration. Run `python test.py --help` for available dataset and metric options.

```bash
python test.py --dataset scivqa --split dev \
  --checkpoint data/scivqa_model/best.pt \
  --report_json results/scivqa/dev_report.json \
  --predictions_csv results/scivqa/dev_predictions.csv
```

Use `--split test` for final held-out evaluation after decisions are complete. Do not tune against test results.

`compare_scivqa.py` compares a prediction CSV with the official split answers. It takes `--split dev|test` and `--predictions_csv`; it does not take checkpoint arguments:

```bash
python compare_scivqa.py --split dev \
  --predictions_csv results/scivqa/dev_predictions.csv \
  --output_json results/scivqa/paper_comparison.json
```

Use the repository's saved reports as the source of exact metric names and values. Include checkpoint, split, dataset revision, and resolved configuration when reporting results. A comparison with a paper is contextual and may differ in model, preprocessing, data, or protocol; report those differences.
