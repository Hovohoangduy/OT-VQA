# OT-VQA Architecture and Source Pipeline

## Active architectures

The runtime supports native Cross-Attention and a general evidence-routing family.
Evidence routing operates on ViT spatial patches plus learned global and null tokens;
it does not require object detections or scene graphs.

```text
image -> frozen ViT -> spatial patches -> evidence bank --------+
                                                               |
question -> frozen BERT -> question tokens -> reasoning slots   |
                                      |                        |
                                      +-> routing costs <-------+
                                               |
                              semi-relaxed OT or softmax control
                                               |
                                      weighted evidence readout
                                               |
                                   slot/question/slot reasoning
                                               |
                                          repeat T steps
                                               |
                                    K decoder-memory tokens
                                               |
                                    autoregressive answer decoder
```

All visual input to an evidence-routing decoder passes through its allocation plan.
Question information enters through slot initialization and slot-to-question attention.

## Evidence bank

`ImageEmbedding.spatial_tokens()` removes the ViT/DeiT prefix tokens. The router:

1. Projects spatial patches to `routing_dim`.
2. Adds checked two-dimensional patch positions and a spatial type embedding.
3. Computes a masked global token from the spatial evidence.
4. Appends a learned null token.

Patch-grid geometry comes from the image-encoder configuration. A mismatch between the
configured grid and the number of cached spatial tokens raises an error. Newly written
cache manifests also record grid size and prefix-token count.

## Question-conditioned slots and cost

For pooled question representation `q` and learned slot identity `e_i`:

```text
S_i(0) = LayerNorm(e_i + Wq q)
```

At each reasoning step, normalized slot queries and evidence keys produce the bounded
cosine cost:

```text
C_ij = -cos(Ws S_i, Wv V_j)
```

The question-conditioned visual preference is smoothed with a uniform distribution over
valid non-null evidence. The predicted null preference is bounded by configuration.
A `uniform` preference option provides an ablation.

## Semi-relaxed OT

For slot marginal `a`, evidence preference `b`, and plan `P`, the core engine solves:

```text
min  <P,C> + epsilon * sum(P * (log(P)-1)) + tau * KL(P^T 1 || b)
 P
s.t. P >= 0 and P 1 = a
```

The row constraint gives each slot a fixed evidence budget. The column marginal is soft,
so irrelevant patches need not receive a prescribed amount. The null column represents
unsupported requests without renormalizing uncertainty away.

The solver uses fixed-count log-domain iterations in float32. It recomputes the final row
dual so returned plans respect the hard marginal. Masked evidence uses a finite internal
log floor to keep backward gradients finite, and its returned plan entries are exactly
zero.

At `tau=0`, the implementation calls the exact independent entropy-regularized row
softmax. `softmax_evidence_routing` uses this same control directly.

## Iterative readout

Each slot receives appearance, spatial moments, and spatial/global/null mass from its
transport row. A recurrent update is followed by slot self-attention, question attention,
and a feed-forward update. The starting configuration shares these weights over two
reasoning steps. Final slots are projected to decoder width and become unmasked decoder
memory.

The training objective is ordinary shifted autoregressive answer cross-entropy. No
retrieval teacher or distillation loss is active for routing models. Gradients pass
through every unrolled transport iteration into cost, visual-preference, evidence, and
slot modules.

## Controls and interpretation

| Model | Purpose |
| --- | --- |
| `cross_attention` | Existing architecture/performance reference |
| `softmax_evidence_routing` | Same evidence and slot reasoner, independent routing |
| `ot_evidence_routing` | Semi-relaxed column-coupled routing |
| OT with `routing_tau=0` | Exact mathematical independent-routing limit |

Only the softmax/OT pair isolates the transport constraint. A difference from native
Cross-Attention also includes the effect of the slot-reasoning architecture.

## Diagnostics

When enabled, each reasoning step measures normalized row entropy, slot-assignment
similarity, null fraction, generalized column KL, hard-row error, fixed-point residual,
finite-plan and convergence rates, iteration count, cost mean/std, and evidence coverage.
The interface reports averages across steps and the final-step value separately.

Evaluation adds generated EM/F1, validation loss, latency, CUDA peak memory, unique
predictions, and top-answer fraction. `diagnose_training.py` works with either routing or
Cross-Attention diagnostics and measures image/question shuffle sensitivity.

## Cached, online, and mixed-precision paths

Online and cached features enter the same `VQAModel._fuse()` path. Frozen encoders remain
in evaluation mode. CUDA mixed precision is opt-in with `--mixed_precision`; the router,
reasoning modules, and decoder can use float16 while the OT solver explicitly disables
autocast and computes in float32.

## Single GPU and DDP

Single GPU runs `python train.py`. Two-GPU training uses:

```bash
torchrun --standalone --nproc_per_node=2 train.py ...
```

DDP uses one replica per GPU, a distributed training sampler, synchronized gradients and
scalar metrics, rank-zero validation/checkpointing, broadcast early-stop decisions, and
clean process-group shutdown. The softmax control keeps matched preference parameters in
the DDP graph with exact zero gradients.

## Checkpoints

| Family | Architecture marker | Format |
| --- | --- | --- |
| Cross-Attention | `cross_attention_only_v1` | v3 |
| OT/softmax evidence routing | `ot_evidence_routing_v1` | v3 |
| Historical teacher resume | Cross-Attention marker | v4 |

Complete fusion/routing configuration is stored in `model_config`. Strict loading checks
that the marker matches the selected fusion. Cached-feature exports may omit frozen
encoder weights and reload them from recorded model identifiers at deployment.

## Source map

| Path | Responsibility |
| --- | --- |
| `model/ot_routing.py` | Routing config, semi-relaxed solver, evidence bank, slot reasoner, diagnostics |
| `model/vqa_model.py` | Architecture construction, shared feature paths, decoder and generation |
| `model/fusion_methods.py` | Native Cross-Attention baseline |
| `model/features_extraction.py` | Frozen encoders, patch grid, answer embeddings |
| `configs/arg_parser.py` | Architecture and routing CLI |
| `train.py` | Direct-answer training, DDP, mixed precision, model selection |
| `utils/checkpoint.py` | Strict architecture-aware persistence |
| `scripts/run_ot_routing_experiment.py` | Matched pilot and confirmation runner |
| `scripts/summarize_ot_routing.py` | Per-seed and paired OT/softmax result summary |
| `test.py`, `predict.py`, `diagnose_training.py` | Evaluation, inference, and reliance diagnostics |
| `tests/test_ot_routing.py` | Solver and router numerical contracts |

## Evidence boundary

The code is implemented and unit-tested. No training result currently demonstrates that
evidence-routing OT improves VQA performance. The required evidence is a paired
multi-seed improvement over the matched softmax model followed by held-out test
confirmation and a measured latency comparison.
