---
name: vqa-evaluation
description: Evaluate OT-VQA checkpoints using supported dataset splits, output reports, and metrics. Use for validation, final test scoring, or prediction comparisons.
---

# OT-VQA evaluation

Use `test.py` for checkpoint evaluation. The supported split values are `dev` and `test`; `dev` is used for model selection and iteration. Check `python test.py --help` and the selected dataset configuration for exact metric behavior.

```bash
python test.py --dataset scivqa --split dev \
  --checkpoint data/scivqa_model/best.pt \
  --report_json results/scivqa/dev_report.json \
  --predictions_csv results/scivqa/dev_predictions.csv
```

Keep the held-out test set for final evaluation after model choices are fixed. Include the checkpoint, dataset revision, split, and resolved run configuration when reporting results.

`compare_scivqa.py` scores a predictions CSV against the local official SciVQA split answers. It does not compare model checkpoints directly:

```bash
python compare_scivqa.py --split dev \
  --predictions_csv results/scivqa/dev_predictions.csv \
  --output_json results/scivqa/dev_paper_comparison.json
```

The script accepts `dev` or `test`; its default gold CSV follows that split. Check the generated report for the exact ROUGE implementation and aggregation. Paper comparisons are contextual because model, data, preprocessing, and evaluation protocol may differ.
