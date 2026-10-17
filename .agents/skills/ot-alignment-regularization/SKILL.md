---
name: ot-alignment-regularization
description: Investigate weak visual grounding or degenerate answer generation in OT-VQA and design controlled alignment or anti-collapse experiments. Use when diagnostics suggest a grounding failure.
---

# OT alignment and answer collapse

Do not assume Optimal Transport automatically prevents answer collapse, and do not infer a root cause from loss values alone. Inspect examples, training and validation metrics, image-replacement behavior, masks, transport marginals, dustbin mass, and saved configuration.

The repository supports an optional auxiliary OT alignment loss (`--ot_loss_weight`) and related OT controls. Their presence does not establish that they improve VQA quality. Check the current implementation in `model/optimal_transport.py`, `model/vqa_model.py`, and the training loss before suggesting a change.

For an intervention:

1. State the observed evidence and one falsifiable hypothesis.
2. Use a fresh `--model_path` and hold other settings fixed.
3. Evaluate on `dev` and compare answer metrics plus grounding/transport diagnostics.
4. Keep `test` held out until selecting a final variant.

Do not prescribe EOS suppression, epsilon schedules, dynamic dustbin gates, or contrastive objectives unless the current code and requested experiment support them. Treat each as a separate architectural or objective change requiring validation.
