# OT-VQA

Train a visual question answering model with SAN, Balanced OT, or question-conditioned
Unbalanced OT fusion. Evaluation and prediction generate answers without references.

```bash
python -m pip install -r requirements.txt
python train.py --model_path data/gqa_model
python test.py --model_path data/gqa_model
python predict.py --checkpoint data/gqa_model/best.pt --image path/to/image.jpg --question "What is in the picture?"
```

This repository supports English text and uses `bert-base-uncased` by default. The first run downloads DeiT and English BERT. Questions and answers receive whitespace normalization before the encoder tokenizer processes them. CSVs require `image`, `question`, and `answer`; `anno_id` is optional. Image paths are relative to `--img_path`. Training needs only its training CSV; evaluation loads only its selected split. Use `--split test --test_csv_path ...` to score a labelled test set. Prediction needs no answer or CSV.

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

Tests create tiny local BERT/DeiT models and do not download pretrained weights. They
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
