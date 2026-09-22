# OT-VQA

Train a visual question answering model with SAN, BAN, MUTAN, cross-attention, or a
lightweight Q-Former, using no OT, Balanced OT, or question-conditioned UOT alignment.
Evaluation and prediction generate answers without references.

```bash
python -m pip install -r requirements.txt
python train.py --model_path data/gqa_model
python test.py --model_path data/gqa_model
python predict.py --checkpoint data/gqa_model/best.pt --image path/to/image.jpg --question "What is in the picture?"
```

This repository uses `google/vit-base-patch16-224-in21k` as its default visual encoder and
`bert-base-uncased` as its default text encoder. The first run downloads both pretrained
models. Questions and answers receive whitespace normalization before the encoder tokenizer
processes them. CSVs require `image`, `question`, and `answer`; `anno_id` is optional. Image
paths are relative to `--img_path`. Training needs only its training CSV; evaluation loads
only its selected split. Use `--split test --test_csv_path ...` to score a labelled test
set. Prediction needs no answer or CSV.

The visual and text encoders can still be overridden with `--image_model` and
`--text_model`.

**Retrain old checkpoints.** The previous code used an incorrect objective and exposed full answers through cross-attention. Corrected training predicts the next token from a shifted answer prefix with causal attention. New version-3 checkpoints include the fusion configuration, optimizer, scheduler, progress, preprocessing settings, and random states. Version-2 SAN checkpoints still load; an old bare state dictionary raises an explanatory error.

Training EM/F1 are teacher-forced diagnostics. `test.py` calculates EM/F1 from autoregressive generation and reports teacher-forced loss separately. No quality claim can be made without retraining and evaluating on real held-out data.

## English GQA dataset

The included subset uses bare image filenames and separate image folders. Use an English text encoder and English preprocessing:

```bash
python train.py --train_csv_path data/gqa_dataset/train.csv --dev_csv_path data/gqa_dataset/val.csv --train_img_path data/gqa_dataset/images/train --dev_img_path data/gqa_dataset/images/val --model_path data/gqa_model
python test.py --dev_csv_path data/gqa_dataset/val.csv --dev_img_path data/gqa_dataset/images/val --model_path data/gqa_model
python predict.py --checkpoint data/gqa_model/best.pt --image data/gqa_dataset/images/test/IMAGE_ID.jpg --question "What color is it?"
```

The downloader writes split-relative image paths such as `train/123.jpg`, so newly downloaded CSVs can use `--img_path data/gqa_dataset/images` for all splits. Its generated corpus contains training text only; the existing corpus predates this correction.

## Optimal Transport fusion

Use `balanced_ot` for exact marginals or `uot` for KL-relaxed marginals. The supplied
CPU and GPU profiles control OT dimension, regularization, and Sinkhorn iterations.

Use `balanced_ot_san` or `uot_san` to add a masked Stacked Attention Network after OT
fusion. OT-SAN prepends a gated global summary to the local OT-fused question tokens, so
the decoder retains the original alignments while gaining a compact global context. The
default OT-SAN configuration uses one layer, hidden dimension 128, dropout 0.2, and gate
logit -2.0. See
[`docs/ot_san_implementation_plan.md`](docs/ot_san_implementation_plan.md) for tensor
contracts, tests, and the controlled experiment plan.

```bash
python train.py \
  --train_csv_path data/gqa_dataset/train.csv \
  --dev_csv_path data/gqa_dataset/val.csv \
  --train_img_path data/gqa_dataset/images/train \
  --dev_img_path data/gqa_dataset/images/val \
  --fusion uot --ot_profile configs/ot_cpu.json \
  --model_path data/gqa_uot --diagnostics

python test.py \
  --dev_csv_path data/gqa_dataset/val.csv \
  --dev_img_path data/gqa_dataset/images/val \
  --checkpoint data/gqa_uot/best.pt --diagnostics

python predict.py \
  --checkpoint data/gqa_uot/best.pt \
  --image data/gqa_dataset/images/test/IMAGE_ID.jpg \
  --question "What color is it?" --diagnostics \
  --diagnostics_output data/gqa_uot/example_transport.png
```

Train the regularized OT-SAN model from scratch on Apple MPS. This command intentionally
has no `--resume` argument and writes to a new experiment directory:

```bash
python train.py \
  --device mps \
  --epochs 50 \
  --batch_size 2 \
  --fusion uot_san \
  --ot_profile configs/ot_mps.json \
  --train_csv_path data/gqa_dataset/train.csv \
  --dev_csv_path data/gqa_dataset/val.csv \
  --img_path data/gqa_dataset/images \
  --d_model 384 \
  --ffn_hidden 1024 \
  --num_layers 2 \
  --num_heads 4 \
  --drop_prob 0.2 \
  --ot_san_layers 1 \
  --ot_san_hidden_dim 128 \
  --ot_san_dropout 0.2 \
  --ot_san_gate_init -2.0 \
  --freeze_answer_embeddings \
  --weight_decay 0.05 \
  --gradient_clip 1.0 \
  --label_smoothing 0.1 \
  --early_stopping_patience 8 \
  --seed 1105 \
  --model_path data/gqa_uot_san_scratch_seed1105 \
  --diagnostics
```

Use `configs/ot_cpu.json` with `--device cpu` or `configs/ot_gpu.json` with
`--device cuda` on other hardware. The selected checkpoint is written to
`data/gqa_uot_san_scratch_seed1105/best.pt`.

Training writes `last.pt` each epoch and updates `best.pt` using generated validation
F1, with validation loss as the tie-breaker. Resume an interrupted run with
`--resume data/gqa_uot/last.pt` and keep `--epochs` set to the total target epoch count.
New runs default to the small-data profile: `d_model=384`, two decoder layers,
`ffn_hidden=1024`, dropout `0.2`, frozen answer embeddings, AdamW weight decay `0.05`,
and gradient clipping at `1.0`. Each setting remains configurable from the CLI.

Frozen encoder features can be cached as float16. Build both split caches under one
root so training can select `train/` and `dev/` automatically:

```bash
python precompute_features.py --csv data/gqa_dataset/train.csv --img_path data/gqa_dataset/images/train --output data/gqa_cache/train --text_model bert-base-uncased
python precompute_features.py --csv data/gqa_dataset/val.csv --img_path data/gqa_dataset/images/val --output data/gqa_cache/dev --text_model bert-base-uncased

python train.py \
  --train_csv_path data/gqa_dataset/train.csv \
  --dev_csv_path data/gqa_dataset/val.csv \
  --feature_cache data/gqa_cache \
  --fusion uot --ot_profile configs/ot_cpu.json \
  --model_path data/gqa_uot_cached
```

The cache manifest fingerprints the CSV and records encoder and preprocessing settings.
Loading refuses stale data or a different encoder. For standalone evaluation, pass the
split cache itself, for example `--feature_cache data/gqa_cache/dev`.

## Fusion methods and benchmark

The implemented comparison covers SAN, BAN, MUTAN, cross-attention Transformer, and
Q-Former with no transport, Balanced OT, and Unbalanced OT (UOT). See
[`docs/fusion_methods_benchmark_plan.md`](docs/fusion_methods_benchmark_plan.md) for the
method definitions, fairness controls, tensor contracts, tests, and experiment milestones.

Fusion names are `san`, `ban`, `mutan`, `cross_attention`, and `qformer`. Prefix a token
fusion with `uot_` or `balanced_ot_`; for example, `uot_ban` and
`balanced_ot_qformer`. Method settings are available through `--ban_glimpses`,
`--ban_dim`, `--mutan_rank`, `--mutan_dim`, `--cross_fusion_layers`,
`--qformer_queries`, `--qformer_layers`, `--qformer_ffn_hidden`, and
`--fusion_dropout`.

Run the script from the repository root. Make it executable once, then check that every
requested fusion name is available without starting training:

```bash
chmod +x scripts/run_fusion_benchmark.sh
DEVICE=mps PREFLIGHT_ONLY=1 scripts/run_fusion_benchmark.sh
```

For a short end-to-end smoke test of all fifteen configurations, use one seed and two
epochs:

```bash
DEVICE=mps \
EPOCHS=2 \
SEEDS="1105" \
METHODS="san ban mutan cross_attention qformer" \
TRANSPORTS="none balanced uot" \
RUN_ROOT=results/fusion_smoke_mps \
scripts/run_fusion_benchmark.sh
```

The complete three-seed no-OT versus Balanced-OT versus UOT benchmark is:

```bash
DEVICE=mps \
OT_PROFILE=configs/ot_mps.json \
METHODS="san ban mutan cross_attention qformer" \
TRANSPORTS="none balanced uot" \
SEEDS="1105 1106 1107" \
scripts/run_fusion_benchmark.sh
```

Use `DEVICE=cpu` or `DEVICE=cuda` to select another backend. The script automatically
selects `configs/ot_cpu.json`, `configs/ot_mps.json`, or `configs/ot_gpu.json`. Override
the profile with `OT_PROFILE=path/to/profile.json` when needed. For example, a short CPU
run of only BAN and MUTAN is:

```bash
DEVICE=cpu METHODS="ban mutan" SEEDS="1105" EPOCHS=2 \
RUN_ROOT=results/fusion_smoke scripts/run_fusion_benchmark.sh
```

To run only selected transport modes, override `TRANSPORTS`. For example, compare
Balanced OT directly with UOT without training the non-OT baseline:

```bash
DEVICE=mps \
TRANSPORTS="balanced uot" \
SEEDS="1105" \
RUN_ROOT=results/fusion_balanced_vs_uot \
scripts/run_fusion_benchmark.sh
```

Dataset locations and other runner settings can be overridden with environment variables:

```bash
TRAIN_CSV=data/gqa_dataset/train.csv \
DEV_CSV=data/gqa_dataset/val.csv \
IMG_PATH=data/gqa_dataset/images \
TEXT_MODEL=bert-base-uncased \
IMAGE_MODEL=google/vit-base-patch16-224-in21k \
BATCH_SIZE=2 \
EPOCHS=50 \
DIAGNOSTICS=1 \
RUN_ROOT=results/fusion_benchmark/my_run \
scripts/run_fusion_benchmark.sh
```

Every run is stored below `RUN_ROOT/<method>/<transport>/seed_<seed>/`. Each directory
contains the exact command, `train.log`, and a `model/` folder containing `best.pt`,
`last.pt`, and `metrics.jsonl`. After all runs finish, the root contains `runs`,
`aggregate`, `paired_deltas`, and `paired_aggregate` reports in both CSV and JSON formats.
The paired reports calculate Balanced-OT minus no-OT, UOT minus no-OT, and UOT minus
Balanced-OT deltas whenever both sides of a comparison are present.

To train SAN and UOT-SAN concurrently on two CUDA GPUs and then produce the same paired
reports, provide the physical GPU IDs through `GPUS`. The runner keeps at most one
training process on each GPU and releases all of a run's device memory when that process
exits:

```bash
DEVICE=cuda \
GPUS="0 1" \
METHODS="san" \
TRANSPORTS="none uot" \
SEEDS="1105 1106 1107" \
EPOCHS=50 \
RUN_ROOT=results/san_vs_uot_2gpu \
scripts/run_fusion_benchmark.sh
```

Set `PARALLEL_WORKERS=1` to retain explicit GPU selection while returning to sequential
execution. `PARALLEL_WORKERS` cannot exceed the number of IDs in `GPUS`.

### Single BERT Loading Optimization

To ensure maximum benchmark efficiency, `scripts/run_fusion_benchmark.sh` (backed by
`scripts/run_fusion_benchmark.py`) loads BERT and ViT **only once** at the beginning. It precomputes
the frozen visual/question feature caches into `FEATURE_CACHE` (`data/gqa_cache` by default) along with
the vocabulary embedding weights (`embeddings.pt`). Each subsequent method, transport, and seed run
consumes cached features directly with `skip_encoders=True`, avoiding reloading the 440MB BERT model
and eliminating redundant forward passes across all 15 configurations.

The runner deliberately refuses to overwrite an existing `RUN_ROOT` and does not resume
old checkpoints. Choose a new directory when rerunning an experiment. If training stops
partway through, completed checkpoints remain available, but launch a new `RUN_ROOT` for
the next controlled benchmark.

## Apple Silicon GPU training with MPS

All entry points accept `--device auto|cpu|cuda|mps`. The default `auto` selects CUDA
first, then Apple MPS, then CPU. Use `--device mps` when you want the command to fail
instead of silently falling back if Metal acceleration is unavailable.

Verify that your PyTorch installation can access the Mac GPU:

```bash
python -c "import torch; print(torch.backends.mps.is_built(), torch.backends.mps.is_available())"
```

Both values should be `True`. Train the OT model with the MPS-specific profile and a
conservative initial batch size:

```bash
python train.py \
  --device mps --batch_size 2 \
  --train_csv_path data/gqa_dataset/train.csv \
  --dev_csv_path data/gqa_dataset/val.csv \
  --train_img_path data/gqa_dataset/images/train \
  --dev_img_path data/gqa_dataset/images/val \
  --fusion uot --ot_profile configs/ot_mps.json \
  --d_model 384 --ffn_hidden 1024 --num_layers 2 \
  --drop_prob 0.2 --freeze_answer_embeddings \
  --weight_decay 0.05 --gradient_clip 1.0 \
  --label_smoothing 0.1 --early_stopping_patience 8 \
  --model_path data/gqa_uot_mps

python test.py --device mps \
  --dev_csv_path data/gqa_dataset/val.csv \
  --dev_img_path data/gqa_dataset/images/val \
  --checkpoint data/gqa_uot_mps/best.pt

python predict.py --device mps \
  --checkpoint data/gqa_uot_mps/best.pt \
  --image data/gqa_dataset/images/test/IMAGE_ID.jpg \
  --question "What color is it?"
```

If an operation is unsupported by the installed PyTorch MPS backend, macOS can run that
operation on CPU while leaving supported operations on the GPU:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py --device mps ...
```

Start without fallback so unsupported operations are visible. Enable it only if PyTorch
reports a specific missing MPS kernel. Reducing `--batch_size` is the first response to
MPS out-of-memory errors. Feature caching further reduces repeated encoder computation.

For small datasets, the compact decoder flags above reduce memorization. Training uses
generated validation F1 for both `best.pt` and early stopping; `last.pt` remains the most
recent state. The training loss uses label smoothing while validation loss remains plain
cross-entropy, so validation loss stays comparable across runs. Fresh runs replace
`metrics.jsonl`; `--resume` appends to it.

Run the bottleneck diagnostic against a checkpoint to measure output collapse, image
reliance, question reliance, OT convergence, and the train/validation gap:

```bash
python diagnose_training.py --device mps \
  --checkpoint data/gqa_uot_mps/best.pt \
  --train_csv_path data/gqa_dataset/train.csv \
  --dev_csv_path data/gqa_dataset/val.csv \
  --dev_img_path data/gqa_dataset/images/val \
  --samples 32
```

The OT device profiles use the convergence settings measured on the GQA checkpoint:
`epsilon=0.1`, tolerance `1e-3`, and up to 50 iterations. Changing the OT profile changes
the learned model, so start a new training run instead of resuming a checkpoint made with
the older profile.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests create tiny local BERT and ViT models and do not download pretrained weights. They
check shifted targets, causal and memory-mask isolation, gradients, Sinkhorn marginals,
UOT mass relaxation, padding, cache validation, generation, and v2/v3 checkpoint loading.

See [the source review](docs/source_review.md) for the problems found and verification details.

See the [Question-Conditioned UOT implementation plan](docs/optimal_transport_implementation_plan.md)
for the design, interfaces, tests, and controlled experiment matrix. The implementation
provides the planned engineering paths; accuracy comparisons still require training the
documented experiment matrix on held-out data.

Open the standalone [visual OT-VQA architecture guide](docs/optimal_transport_vqa_architecture.html)
for the full component flow, tensor shapes, equations, training and inference paths,
interactive Sinkhorn intuition, caching, checkpoints, and transport diagnostics.
