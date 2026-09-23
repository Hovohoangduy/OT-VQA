# Simplified OT-VQA Architecture and Source Pipeline

## Scope

The active repository contains one deployable VQA architecture: native
Cross-Attention. Optimal Transport is a training-only research component. No transport
plan, OT projection, or Sinkhorn iteration executes in `test.py` or `predict.py`.

The following were removed because they were either ineffective in the recorded
experiments or irrelevant to the next OT-teacher investigation:

- SAN and OT-SAN;
- BAN and MUTAN;
- Q-Former;
- gated OT-aligned Cross-Attention;
- runtime Balanced OT/UOT fusion and log-attention priors;
- transport visualization and hardware OT profiles;
- the 15-configuration fusion benchmark runner;
- legacy fusion checkpoint migration.

This is an intentional compatibility break. Only checkpoints written by the simplified
`cross_attention_only_v1` architecture load.

## Deployed VQA graph

```text
image [B,3,H,W]
  └─ frozen ViT/DeiT
       └─ remove CLS/distillation prefix
            └─ visual patches V [B,N,Dv]

question strings
  └─ frozen English BERT
       └─ remove PAD and special tokens through mask
            └─ question tokens Q [B,M,Dq]

V ───────────────────────────────┐
                                 ├─ CrossAttentionFusion
Q ── query projection ───────────┘     Q queries; V keys/values
                                            │
                                            ▼
                                  decoder memory H [B,M,d]
                                            │
answer prefix [BOS,y1,...] ─ embeddings ─ causal Transformer decoder
                                            │
                                            ▼
                                      next-token logits
```

### Cross-Attention

For every head:

```text
Qh = Wq Q
Kh = Wk V
Vh = Wv V
A  = softmax(Qh Khᵀ / sqrt(d_head))
Z  = A Vh
```

The layer applies residual projection, LayerNorm, a feed-forward network, another
residual, and another LayerNorm. Padded visual patches are blocked before softmax.
Padded question positions are zeroed after attention and after the feed-forward update.

`FusionOutput` contains:

- `memory`: contextual question tokens consumed by the answer decoder;
- `memory_padding_mask`: valid decoder-memory positions;
- optional final-layer attention `[B, heads, question, visual]` for diagnostics or OT
  distillation;
- optional attention entropy and memory statistics.

## Training-only OT teacher

The teacher is not a fusion method. It is an auxiliary model used only when
`--alignment_mode ot_contrastive_distill` is enabled.

### Stage 1: contrastive alignment

1. `extract_alignment_features()` obtains frozen ViT patches and BERT question tokens.
2. Independent trainable adapters apply `Linear → GELU → LayerNorm → L2 normalization`.
3. Pairwise cosine cost is `Cij = 1 - cosine(vi, qj)`.
4. Uniform masked marginals and float32 log-domain UOT produce plan `P`.
5. Hard mismatched pairs are selected with pooled adapter similarity.
6. Symmetric InfoNCE trains image-to-question and question-to-image ranking.
7. A detached FIFO queue supplies candidates for small batches.

The current pair score is:

```text
score(V,Q) = -sum(P * C) / max(sum(P), minimum_mass)
```

This score is the principal remaining research weakness: normalizing away matched mass
can allow an unrelated pair to obtain a competitive score using a small amount of
low-cost transport. The next improvement should replace it with a global-to-local
objective and a complete regularized UOT energy; that work is deliberately not hidden
inside this cleanup.

### Validation gate

The teacher may supervise VQA only when:

- mean positive score exceeds the hardest-negative score;
- I2Q retrieval exceeds `1 / candidate_count`;
- Q2I retrieval exceeds `1 / candidate_count`;
- transport diagnostics and feature variances are finite and non-collapsed.

If the gate fails:

- `error` stops immediately;
- `fallback` trains native Cross-Attention with KL and OT weight equal to zero.

A fallback result is a native result, not an OT result.

### Stage 2: attention distillation

After a successful gate, the teacher is frozen. Its positive plan is normalized over
visual patches for each valid question token and detached. The student minimizes:

```text
Ltotal = Lanswer + lambda(epoch) * KL(stopgrad(Pword→patch) || mean_heads(A))
```

The answer loss remains primary. `lambda(epoch)` warms up from zero to
`--ot_distill_weight`.

## Answer training and generation

Training uses shifted targets:

```text
decoder input  = [BOS, y1, y2, ...]
target         = [y1,  y2, EOS, ...]
```

The decoder uses causal self-attention and cross-attention to fused memory. Training
cross-entropy ignores PAD and may use label smoothing. Checkpoints are selected with
autoregressively generated validation F1, using validation loss as a tie-breaker.

Inference begins with BOS, greedily appends a token, stops each row at EOS, and pads
finished rows independently. It never receives the reference answer.

## Cached and online feature paths

- Online mode runs frozen ViT and BERT inside the model.
- Cached mode loads float16 encoder tokens and bypasses both backbones during training.
- The cache manifest verifies CSV fingerprint, encoder identity, split, and preprocessing.
- `VQAModel.encode_from_features()` and the online path share the same fusion and decoder.

## Single and two-GPU execution

Single GPU uses `python train.py`. Two GPUs use:

```bash
torchrun --standalone --nproc_per_node=2 train.py ...
```

DDP behavior:

- one process and model replica per GPU;
- `DistributedSampler` creates non-overlapping training shards;
- student and teacher gradients are all-reduced;
- scalar training metrics are reduced;
- rank zero evaluates the complete development split and writes checkpoints;
- gate, fallback, and early-stop decisions are broadcast;
- process groups are destroyed during normal completion and exceptions.

## Checkpoints

Every new checkpoint contains `architecture=cross_attention_only_v1`.

| File | Format | Contents |
| --- | --- | --- |
| Native `last.pt` / `best.pt` | v3 | student, architecture, optimizer/scheduler, progress, RNG |
| OT run `last_training.pt` | v4 | v3 fields plus teacher, queue, alignment config and stage |
| OT run `best.pt` | v3 | deployable student only |

Legacy fusion and pre-cleanup state layouts are rejected explicitly.

## Source map

| Path | Responsibility |
| --- | --- |
| `configs/arg_parser.py` | One fusion choice plus training-only OT arguments |
| `model/features_extraction.py` | Frozen ViT/BERT features and answer embeddings |
| `model/fusion_methods.py` | Cross-Attention configuration, masks, layers and diagnostics |
| `model/vqa_model.py` | Feature routing, fusion, decoding and generation |
| `model/optimal_transport.py` | Minimal float32 Sinkhorn kernel used only by the teacher |
| `model/ot_alignment.py` | Adapters, negative queue, UOT contrastive teacher and KL target |
| `utils/ot_alignment_training.py` | Warm-up, validation gate and distillation epochs |
| `utils/distributed.py` | DDP initialization, reductions, broadcasts and cleanup |
| `utils/checkpoint.py` | v3/v4 persistence and architecture compatibility checks |
| `train.py` | Native or staged training orchestration |
| `test.py` | Generated EM/F1, loss, latency and output diversity |
| `predict.py` | Single-example answer generation and attention diagnostics |
| `diagnose_training.py` | Modality shuffles, output collapse and attention entropy |

## Evidence boundary

The failed teacher log supplied on 2026-09-23 ended with margin `-0.0421`, I2Q `0.246`,
and Q2I `0.219` for four candidates. It correctly activated fallback; every VQA epoch
used `KL=0` and weight `0`. Its best F1 of `0.3323` is therefore not an OT result.

No accuracy improvement is claimed by this cleanup. It establishes a smaller, auditable
baseline for the next teacher redesign.
