# OT-VQA repository guidance

## Project goal

This repository develops Visual Question Answering with Optimal Transport as the core multimodal fusion engine. The supported model uses question-conditioned partial OT between spatial image features and valid question tokens, then autoregressively generates an answer. Preserve that design intent when changing the model; do not introduce a non-OT fusion path as the default.

## Source of truth

- OT solver and fusion: `model/optimal_transport.py`
- VQA model integration: `model/vqa_model.py`
- Training and evaluation behavior: `train.py`, `test.py`, and `configs/`
- Current usage: `README.md`; implementation: code/config/checkpoint. Treat `SCIVQA_TUNING.md` as historical experiment notes and verify every recipe against the current CLI and `configs/config.py`.

Read the relevant code and docs for the task. Treat implementation and saved checkpoint configuration as authoritative where older skill examples differ.

## Skill routing

Repo-local Codex skills live in `.agents/skills/<skill-name>/SKILL.md`. Load the smallest relevant set for the task:

- OT math, marginals, Sinkhorn, dustbins, transport plans: `optimal-transport`
- New fusion designs or research hypotheses: `ot-architecture-design`
- OT throughput, memory, or numerical stability: `ot-performance-optimization` and `performance-profiling`
- Alignment collapse or auxiliary OT objectives: `ot-alignment-regularization` and `training-diagnostics`
- PyTorch implementation: `pytorch-patterns`; Python structure: `python-patterns`
- Dataset metrics and split policy: `vqa-evaluation`; controlled studies: `ot-vqa-ablation`
- Training launch/resume: `ot-vqa-training`; run instability: `ot-vqa-diagnostics`
- Evaluation commands: `ot-vqa-evaluation`; safe change validation: `ot-vqa-testing`
- Hard-to-localize failures: `systematic-debugging`
- Reviewing a proposed implementation: `code-review-checklist`

Skills are task guidance, not a requirement to load every file for every change. `.agent/` is a Gemini-era archive; Codex should use this file and `.agents/skills/` as its source of truth.

## Modeling and research rules

- Keep the OT path device-agnostic and compatible with the repository's CUDA, MPS, and CPU support.
- Preserve tensor and mask semantics through cost construction, marginals, Sinkhorn, dustbin handling, transport aggregation, and decoder memory.
- Compare architectural or hyperparameter changes with controlled validation experiments. Keep the test split held out until final evaluation.
- Describe accuracy or quality gains only when supported by measured results. A plausible mechanism or faster runtime alone does not establish a VQA improvement.
- Avoid changing dataset splits, training objectives, or default model architecture unless the request calls for it.

## Working conventions

- Respond in Vietnamese when the user writes in Vietnamese; keep code identifiers and comments in English.
- Make focused edits and preserve existing checkpoint compatibility unless a task explicitly requires a migration.
- Use the repository's existing command-line entry points and configuration rather than duplicating pipeline logic.
- Run checks that are appropriate to the requested change and report exactly what was run. Do not delete project tests or generated research artifacts as cleanup.
