# OT-VQA

The repository provides two VQA architecture families:

- Native Cross-Attention, the established performance baseline.
- Question-conditioned evidence routing, where all visual evidence reaching the
  answer decoder passes through either semi-relaxed Optimal Transport or its matched
  independent-softmax control.

The evidence-routing model supports general patch-level visual evidence. It does not
require object detection or scene graphs. Its implementation is complete, but its VQA
accuracy has not yet been established by training experiments.

## Architecture

```text
image -> frozen ViT patches -> spatial/global/null evidence -----------+
                                                                     |
question -> frozen BERT -> question-conditioned reasoning slots       |
                                |                                    |
                                +-- cost matrix -- semi-relaxed OT <--+
                                           |
                                  OT-weighted evidence
                                           |
                                     update slots
                                           |
                                  repeat reasoning step
                                           |
                                  answer decoder memory
```

Semi-relaxed OT enforces a fixed budget for every reasoning slot and softly matches
the combined slot allocation to a question-conditioned visual preference. The null
token represents unavailable or unnecessary evidence. The solver uses float32
log-domain iterations, including during mixed-precision training.

Available `--fusion` values:

| Value | Purpose |
| --- | --- |
| `cross_attention` | Native baseline |
| `softmax_evidence_routing` | Matched slot reasoner with independent routing |
| `ot_evidence_routing` | Slot reasoner with semi-relaxed OT |

Setting `--fusion ot_evidence_routing --routing_tau 0` gives the exact independent
row-softmax transport limit at the same temperature and slot budget.

## Install

```bash
python -m pip install -r requirements.txt
```

The default encoders are `google/vit-base-patch16-224-in21k` and
`bert-base-uncased`.

## Train the OT model

Single GPU:

```bash
python train.py \
  --device cuda \
  --fusion ot_evidence_routing \
  --alignment_mode none \
  --mixed_precision \
  --routing_slots 4 \
  --routing_steps 2 \
  --routing_dim 256 \
  --routing_epsilon 0.1 \
  --routing_tau 0.5 \
  --routing_iterations 20 \
  --epochs 100 \
  --batch_size 4 \
  --train_csv_path data/gqa_dataset/train.csv \
  --dev_csv_path data/gqa_dataset/val.csv \
  --img_path data/gqa_dataset/images \
  --model_path results/ot_evidence_routing/seed_42 \
  --seed 42 \
  --diagnostics
```

Two T4 GPUs in one Kaggle notebook cell:

```bash
!CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  --device cuda \
  --fusion ot_evidence_routing \
  --alignment_mode none \
  --mixed_precision \
  --routing_slots 4 \
  --routing_steps 2 \
  --routing_dim 256 \
  --routing_epsilon 0.1 \
  --routing_tau 0.5 \
  --routing_iterations 20 \
  --epochs 100 \
  --batch_size 4 \
  --train_csv_path /kaggle/input/datasets/duytrain02/gqa-dataset-1909/gqa_dataset/train.csv \
  --dev_csv_path /kaggle/input/datasets/duytrain02/gqa-dataset-1909/gqa_dataset/val.csv \
  --img_path /kaggle/input/datasets/duytrain02/gqa-dataset-1909/gqa_dataset/images \
  --model_path results/ot_evidence_routing/seed_42 \
  --seed 42 \
  --diagnostics
```

`--batch_size` is per GPU, so the second command has a global batch size of eight.
DDP shards the training set, synchronizes gradients and metrics, validates on rank
zero, and writes each checkpoint once.

## Matched pilot experiment

The experiment runner executes the native baseline, slot-softmax control, OT model,
and the OT `tau=0` limit:

```bash
python scripts/run_ot_routing_experiment.py \
  --train_csv data/gqa_dataset/train.csv \
  --dev_csv data/gqa_dataset/val.csv \
  --img_path data/gqa_dataset/images \
  --output_root results/ot_routing_pilot \
  --seeds 42 \
  --epochs 100 \
  --batch_size 4 \
  --gpus 1
```

Use `--gpus 2` for one two-GPU DDP training job at a time. Use `--dry_run` to print
commands without starting training. Confirmation runs should use seeds 1105, 1106,
and 1107 after the pilot is numerically sound.

Summarize completed runs and paired OT-minus-softmax differences:

```bash
python scripts/summarize_ot_routing.py --root results/ot_routing_pilot
```

The softmax and OT routing models have the same evidence, slots, reasoning updates,
decoder, and parameter layout. A performance difference between native Cross-Attention
and OT does not isolate OT; the matched softmax comparison does.

## Feature cache

Precompute frozen features for each split:

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

Add `--feature_cache data/gqa_cache` to a training command. Cache manifests include
the visual source shape, patch-grid geometry, prefix-token count, encoder identities,
preprocessing, and CSV fingerprint. Existing valid version-1 caches remain readable;
the model also checks the spatial token count against encoder grid metadata.

## Evaluation, prediction, and diagnosis

```bash
python test.py \
  --checkpoint results/ot_evidence_routing/seed_42/best.pt \
  --dev_csv_path data/gqa_dataset/val.csv \
  --img_path data/gqa_dataset/images \
  --split dev --diagnostics

python predict.py \
  --checkpoint results/ot_evidence_routing/seed_42/best.pt \
  --image path/to/image.jpg \
  --question "What is happening in this image?" \
  --diagnostics

python diagnose_training.py \
  --checkpoint results/ot_evidence_routing/seed_42/best.pt \
  --dev_csv_path data/gqa_dataset/val.csv \
  --dev_img_path data/gqa_dataset/images
```

Routing diagnostics include normalized row entropy, slot similarity, null fraction,
visual-marginal KL, row and fixed-point residuals, finite/convergence rates, cost
statistics, iterations, and evidence coverage. Evaluation also reports generated
EM/F1, loss, latency, peak CUDA memory, prediction diversity, and majority rate.

## Checkpoints

- Cross-Attention checkpoints use `architecture=cross_attention_only_v1`.
- OT and matched-softmax checkpoints use `architecture=ot_evidence_routing_v1`.
- Version 3 stores deployable model and standard training state.
- Version 4 remains reserved for the older training-only alignment teacher workflow.
- Strict loading rejects architecture/configuration mismatches.
- A routing initialization may be copied between OT and softmax controls when every
  architecture field except routing mode matches.

## Historical training-only OT teacher

`--fusion cross_attention --alignment_mode ot_contrastive_distill` remains available
for reproducing the earlier contrastive-teacher experiment. It is incompatible with
the evidence-routing architectures. The recorded teacher failed its validation gate,
so it provides no demonstrated OT performance gain.

## Verification and evidence boundary

```bash
python -m compileall -q configs model utils scripts train.py test.py predict.py tests
python -m unittest discover -s tests -q
```

Tests verify the hard slot marginal, padding, finite gradients, exact `tau=0` limit,
router integration, cached/online agreement, checkpoints, generation, and the existing
baseline and teacher contracts. Passing tests establishes software behavior, not VQA
accuracy. Use the matched multi-seed experiment and held-out test split before claiming
that OT improves performance.

See [the implementation plan](docs/ot_evidence_routing_plan.md),
[the source architecture](docs/architecture_and_pipeline.md), and
[the HTML architecture guide](docs/optimal_transport_vqa_architecture.html).
