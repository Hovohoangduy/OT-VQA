# OT-VQA

This repository now contains one VQA architecture: **native Cross-Attention**. Optimal
Transport is retained only as an optional training-time alignment teacher. It is never
part of the deployed model and never adds inference latency.

Removed architectures include SAN, OT-SAN, BAN, MUTAN, Q-Former, gated OT-aligned
Cross-Attention, barycentric runtime fusion, and Balanced/UOT attention priors. Their CLI
options, benchmark runner, profiles, tests, and checkpoint compatibility were also
removed.

## Architecture

```text
image ── frozen ViT ── patch tokens ─────────────┐
                                                 ├─ native Cross-Attention ── memory
question ── frozen BERT ── content tokens ───────┘                         │
                                                                            ▼
answer prefix ── BERT token embeddings ── causal Transformer decoder ── answer
```

Optional training-only OT:

```text
frozen ViT/BERT features ── UOT contrastive teacher ── validation gate
                                                        │
                         pass: soft attention target ───┤
                         fail: native VQA fallback ─────┘
```

The teacher must obtain a positive hard-negative margin and retrieval above chance. If
it fails, the default policy trains native Cross-Attention with OT weight zero. Such a
run is a fallback baseline and is not evidence of an OT improvement.

## Install

```bash
python -m pip install -r requirements.txt
```

Default encoders are `google/vit-base-patch16-224-in21k` and
`bert-base-uncased`. The first online run downloads them.

## Native Cross-Attention baseline

```bash
python train.py \
  --device cuda \
  --fusion cross_attention \
  --epochs 100 \
  --batch_size 4 \
  --train_csv_path data/gqa_dataset/train.csv \
  --dev_csv_path data/gqa_dataset/val.csv \
  --img_path data/gqa_dataset/images \
  --model_path results/cross_attention \
  --seed 42
```

`--fusion cross_attention` is optional but retained so experiment records remain
explicit. No other fusion value is accepted.

## Training-only OT experiment

```bash
python train.py \
  --device cuda \
  --fusion cross_attention \
  --alignment_mode ot_contrastive_distill \
  --alignment_warmup_epochs 10 \
  --ot_alignment_lr 0.0001 \
  --ot_alignment_dim 128 \
  --ot_alignment_iterations 20 \
  --ot_negative_count 3 \
  --ot_negative_queue_size 32 \
  --ot_contrastive_temperature 0.07 \
  --ot_distill_weight 0.02 \
  --ot_distill_warmup_epochs 5 \
  --ot_gate_failure_policy error \
  --epochs 100 \
  --batch_size 4 \
  --train_csv_path data/gqa_dataset/train.csv \
  --dev_csv_path data/gqa_dataset/val.csv \
  --img_path data/gqa_dataset/images \
  --model_path results/ot_distilled_cross_attention \
  --seed 42
```

Use `--ot_gate_failure_policy error` while improving the teacher so an invalid teacher
does not consume time on a long fallback run. Use `fallback` only when completing a
native baseline is useful.

## Two T4 GPUs in a Kaggle notebook

The batch size is per GPU. This command gives a global batch of eight:

```bash
!CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  --device cuda \
  --fusion cross_attention \
  --alignment_mode ot_contrastive_distill \
  --alignment_warmup_epochs 10 \
  --ot_gate_failure_policy error \
  --epochs 100 \
  --batch_size 4 \
  --train_csv_path /kaggle/input/datasets/duytrain02/gqa-dataset-1909/gqa_dataset/train.csv \
  --dev_csv_path /kaggle/input/datasets/duytrain02/gqa-dataset-1909/gqa_dataset/val.csv \
  --img_path /kaggle/input/datasets/duytrain02/gqa-dataset-1909/gqa_dataset/images \
  --model_path results/ot_distilled_cross_attention \
  --seed 42
```

DDP shards training data, synchronizes gradients and metrics, performs full validation
on rank zero, writes checkpoints once, and broadcasts gate/early-stop decisions.

## Feature cache

Precompute frozen features once:

```bash
python precompute_features.py \
  --csv data/gqa_dataset/train.csv \
  --img_path data/gqa_dataset/images \
  --output data/gqa_cache/train

python precompute_features.py \
  --csv data/gqa_dataset/val.csv \
  --img_path data/gqa_dataset/images \
  --output data/gqa_cache/dev
```

Then add `--feature_cache data/gqa_cache` to either training command. A complete cache
root must contain valid `train/` and `dev/` manifests; an empty directory is rejected.

## Evaluation and prediction

```bash
python test.py \
  --checkpoint results/cross_attention/best.pt \
  --dev_csv_path data/gqa_dataset/val.csv \
  --img_path data/gqa_dataset/images \
  --split dev --diagnostics

python predict.py \
  --checkpoint results/cross_attention/best.pt \
  --image path/to/image.jpg \
  --question "What color is the cup?" \
  --diagnostics
```

Diagnostics report Cross-Attention entropy, memory statistics, latency, prediction
diversity, and majority-answer rate. Runtime transport maps no longer exist.

## Checkpoints

- Version 3: deployable Cross-Attention student and ordinary training state.
- Version 4: resumable staged state containing the student, OT teacher, negative queue,
  optimizer, scheduler, RNG state, and current stage.
- `best.pt` from an OT-distillation run is version 3 and contains only the student.
- Legacy fusion and pre-simplification checkpoints intentionally fail with a clear
  retraining message.

## Verification

```bash
python -m unittest discover -s tests -q
```

The focused suite covers native fusion shapes/masks/gradients, autoregressive generation,
raw/cached feature agreement, checkpoint export/resume contracts, Sinkhorn numerics,
contrastive teacher gradients, collapse detection, the decision gate, and attention
distillation.

See [architecture_and_pipeline.md](docs/architecture_and_pipeline.md) for the source
walkthrough, [ot_contrastive_distillation_plan.md](docs/ot_contrastive_distillation_plan.md)
for the remaining OT research plan, and
[optimal_transport_vqa_architecture.html](docs/optimal_transport_vqa_architecture.html)
for the visual architecture guide.
