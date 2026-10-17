---
name: ot-vqa-ablation
description: Design controlled validation ablations for OT-VQA fusion, Sinkhorn, preprocessing, and training settings. Use when testing whether an OT change improves quality or efficiency.
---

# OT-VQA ablation

State a hypothesis and vary one factor at a time. Inspect `configs/config.py`, `configs/arg_parser.py`, and `python train.py --help` before selecting settings. Supported OT controls include `--ot_epsilon`, `--ot_iterations`, `--ot_dustbin_mass`, `--ot_dustbin_cost`, `--ot_num_heads`, `--ot_loss_weight`, `--fgw_alpha`, and `--unbalanced_tau`.

Give every run a fresh `--model_path`; keep data, seed, model, and all non-target settings fixed. Score variants on the same validation split with `test.py --split dev`, then compare reports. Do not use unsupported flags such as `--save-dir`; repository options use underscores and training outputs use `--model_path`.

Report the baseline, variant, exact settings, seed, validation metrics, runtime/memory if efficiency is the target, and limitations. Hold the test split until the final selected variant. Do not infer improved VQA quality from faster execution alone.
