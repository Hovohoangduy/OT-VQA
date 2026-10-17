---
name: ot-vqa-training
description: Prepare or resume OT-VQA training runs with the repository's dataset presets, checkpoint paths, and device options. Use when launching or configuring training.
---

# OT-VQA training

Use the existing `train.py` pipeline and `configs/config.py` presets. Partial Optimal Transport is the supported fusion path (`--fusion ot`). Read the selected dataset preset and current CLI before choosing overrides.

## Start a new run

For SciVQA or PlantExpertVQA, the dataset preset provides data paths and `model_path`. To keep outputs separate, set a fresh `--model_path` for each experiment:

```bash
python train.py --dataset scivqa --model_path data/scivqa_model_run1
python train.py --dataset plantexpert --model_path data/plantexpert_model_run1 --device auto
```

CLI option names use underscores, such as `--batch_size`, `--model_path`, and `--ot_epsilon`. Run `python train.py --help` for the current complete option list. GQA requires explicit data paths if no preset supplies the local files.

## Resume

Use the same output directory and schedule settings as the original run, with its saved checkpoint:

```bash
python train.py --dataset scivqa --model_path data/scivqa_model --resume data/scivqa_model/last.pt
```

Check the checkpoint/config inheritance behavior in `README.md` before changing selection metric, preprocessing, visual memory, or loss normalization during resume.

## Research discipline

- Record the command, seed, dataset revision, checkpoint directory, and resolved config for each run.
- Change one experimental factor at a time when comparing variants.
- Select models on validation metrics. Keep the test split for final evaluation.
- Do not report a VQA quality gain until it is measured against a comparable baseline.
