# Fusion Methods With and Without Optimal Transport

> Implementation status: the five native, balanced-OT, and UOT triplets,
> shared fusion contract, CLI configuration, diagnostics, tests, and automated runner
> are implemented. The smoke and controlled benchmark milestones still require actual
> training runs; no performance conclusion is implied by implementation completion.
> The shared `OptimalTransportFusion` remains intact rather than being split into two
> stateful modules, preserving strict loading of existing version-3 OT checkpoints; new
> methods consume its plan and barycentric tokens through `TransportOutput`.

## 1. Objective

Implement and compare five visual-question fusion families:

1. Stacked Attention Network (SAN)
2. Bilinear Attention Network (BAN)
3. MUTAN
4. Cross-attention Transformer
5. Q-Former

Each family has a non-OT variant, a Balanced Optimal Transport variant, and a
question-conditioned Unbalanced Optimal Transport (UOT) variant.

The primary research question is:

> Under the same encoders, decoder, data split, optimization settings, and random seeds,
> does explicit OT alignment improve generated-answer accuracy or visual grounding for a
> given fusion family?

This plan separates engineering completion from empirical claims. Implementing a method
does not establish that it improves VQA performance.

## 2. Experiment matrix and names

Use these checkpoint-visible fusion names:

| Fusion family | Without OT | With UOT | Optional Balanced OT |
| --- | --- | --- | --- |
| SAN | `san` | `uot_san` | `balanced_ot_san` |
| BAN | `ban` | `uot_ban` | `balanced_ot_ban` |
| MUTAN | `mutan` | `uot_mutan` | `balanced_ot_mutan` |
| Cross-attention Transformer | `cross_attention` | `uot_cross_attention` | `balanced_ot_cross_attention` |
| Q-Former | `qformer` | `uot_qformer` | `balanced_ot_qformer` |

The default automated benchmark runs all fifteen non-OT, Balanced-OT, and UOT
combinations. With three seeds this produces 45 runs.

Keep the existing `san`, `balanced_ot`, `uot`, `balanced_ot_san`, and `uot_san` checkpoint
names loadable. The current plain `balanced_ot` and `uot` names represent OT barycentric
fusion without an additional named fusion backend and remain useful reference baselines.

## 3. Fair-comparison rules

All paired comparisons must keep the following fixed:

- train, validation, and held-out test rows;
- image and text encoder names and revisions;
- tokenizer and image preprocessing;
- frozen/unfrozen encoder policy;
- answer embedding policy;
- `d_model`, decoder depth, heads, and FFN dimension;
- optimizer, scheduler, learning rate, weight decay, and gradient clipping;
- dropout and label smoothing;
- batch size and epoch budget;
- early-stopping rule and patience;
- decoding strategy and maximum answer length;
- seed;
- OT profile across all UOT methods.

The only intended paired difference is the addition of the OT plan and its method-specific
injection. Parameter counts will not be exactly equal because OT adds projections,
marginals, and cost networks. Log both total and trainable counts and report the delta.

Use generated validation F1 for checkpoint selection, with validation loss as the existing
tie-breaker. Never select a method using the held-out test split.

## 4. Shared software contract

### 4.1 Normalized fusion specification

Parse every user-visible fusion string once:

```python
@dataclass(frozen=True)
class FusionSpec:
    method: str       # san, ban, mutan, cross_attention, qformer, barycentric
    transport: str    # none, balanced, unbalanced

    @property
    def uses_ot(self) -> bool: ...
```

Examples:

```text
san                    -> method=san, transport=none
uot_san                -> method=san, transport=unbalanced
balanced_ot_ban        -> method=ban, transport=balanced
uot                    -> method=barycentric, transport=unbalanced
```

Replace scattered checks such as `fusion_type != "san"` with `fusion_spec.uses_ot`.
This is required because `ban`, `mutan`, `cross_attention`, and `qformer` are non-OT
models even though their names are not `san`.

### 4.2 Fusion input

Every fusion module receives a structured input:

```python
@dataclass
class FusionInput:
    visual_tokens: torch.Tensor             # [B, N, Dv]
    question_tokens: torch.Tensor           # [B, M, Dq]
    visual_padding_mask: torch.Tensor       # [B, N], True means padding
    question_padding_mask: torch.Tensor     # [B, M], True means padding
    transport: Optional[TransportOutput]    # present only for OT variants
```

The encoder must remove ViT/DeiT prefix tokens before constructing `FusionInput`. Question
boundary/padding policy must match the existing OT path.

### 4.3 Fusion output

Every fusion method returns the same decoder-facing contract:

```python
@dataclass
class FusionOutput:
    memory: torch.Tensor                    # [B, L, d_model]
    memory_padding_mask: torch.Tensor       # [B, L]
    diagnostics: Optional[dict[str, torch.Tensor]]
```

Rules:

- `memory` must remain a sequence; do not hide an unmasked padded position.
- `memory_padding_mask.shape` must equal `memory.shape[:2]`.
- padded memory values should be zero where practical and must never affect the decoder;
- diagnostics must not alter logits in evaluation mode;
- all methods must support online and cached frozen-encoder features;
- all methods must support teacher-forced training and reference-free generation.

### 4.4 Registry

Create a single registry/factory rather than adding branches throughout `VQAModel`:

```python
FUSION_REGISTRY = {
    "san": SANFusion,
    "ban": BANFusion,
    "mutan": MUTANFusion,
    "cross_attention": CrossAttentionFusion,
    "qformer": QFormerFusion,
}
```

The factory receives `FusionSpec`, dimensions, OT configuration, and method configuration.
The complete normalized specification must be stored in `model.model_config`.

## 5. Common encoder and OT paths

### 5.1 Non-OT path

```text
image -> ViT spatial tokens V
question -> contextual token features Q
V, Q, masks -> selected native fusion method -> FusionOutput -> answer decoder
```

### 5.2 UOT path

```text
V, Q -> shared OT projections and learned marginals
     -> hybrid cost
     -> log-domain UOT Sinkhorn plan P
P + V + Q -> method-specific OT augmentation
          -> FusionOutput -> same answer decoder
```

All UOT variants use the same `OTConfig` and `OptimalTransportFusion` components. Do not
copy the Sinkhorn implementation into method modules.

The current `OptimalTransportFusion` combines transport computation and a particular
barycentric MLP. Refactor it in two backward-compatible stages:

1. `OptimalTransportAlignment`: projections, marginals, cost, and transport plan.
2. `BarycentricTokenFusion`: aligned visual evidence and current four-way feature MLP.

Keep state-dictionary migration or legacy construction paths so existing version-3 OT and
OT-SAN checkpoints continue to load strictly.

## 6. Method designs

### 6.1 Stacked Attention Network

#### Without OT: `san`

Preserve the existing compatibility path:

```text
global question context -> repeated attention over visual tokens -> one summary token
```

#### With OT: `uot_san`

Preserve the implemented OT-SAN design:

```text
UOT barycentric question tokens H
    -> masked stacked attention over H
    -> gated global summary S
    -> decoder memory [S, H]
```

This is a system-level comparison, not a perfectly parameter-matched SAN ablation: the
legacy non-OT SAN emits one memory token, while OT-SAN retains local tokens. Report this
limitation explicitly. An optional strict ablation may add a `token_san` non-OT variant
that builds dense soft-aligned question tokens without Sinkhorn.

### 6.2 Bilinear Attention Network

BAN uses multiple bilinear attention glimpses over visual and question tokens.

For glimpse `g`:

```text
score_g(i,j) = w_g^T(tanh(Wv_g V_i) * tanh(Wq_g Q_j))
```

Mask invalid `(i,j)` pairs before normalization. Pool a bilinear joint representation per
glimpse and combine glimpses with residual updates. Project the resulting token sequence
to `d_model`.

#### Without OT: `ban`

Normalize native bilinear scores over valid visual-question pairs.

#### With OT: `uot_ban`

Use the OT plan as an additive log prior, not as a hard mask:

```text
augmented_score_g = score_g + lambda_ot * log(P + minimum_mass)
```

Make `lambda_ot` a learned scalar initialized to `1.0`, and report it in diagnostics. This
keeps BAN trainable while making the explicit OT alignment available to every glimpse.

Initial configuration:

```text
glimpses=2, bilinear_dim=256, dropout=0.2
```

### 6.3 MUTAN

Implement a low-rank Tucker-style multimodal fusion without materializing a full
three-dimensional core tensor:

```text
v_r = Wv_r(v)
q_r = Wq_r(q)
z = Wo(sum_r(v_r * q_r))
```

Use `rank=5` initially. Apply dropout to projected factors and LayerNorm after output
projection.

#### Without OT: `mutan`

Learn native question-conditioned visual attention, pool one visual vector per valid
question token, and apply MUTAN to each `(visual_evidence_j, Q_j)` pair.

#### With OT: `uot_mutan`

Replace native visual pooling weights with normalized OT columns:

```text
visual_evidence_j = sum_i(P_ij * V_i) / max(sum_i(P_ij), minimum_mass)
```

The Tucker fusion block and its parameters remain identical between the pair.

Initial configuration:

```text
rank=5, factor_dim=256, dropout=0.2
```

### 6.4 Cross-attention Transformer

Use question tokens as queries and visual tokens as keys/values, followed by a residual
FFN. Preserve the question sequence length as decoder memory.

#### Without OT: `cross_attention`

Use standard masked multi-head cross-attention:

```text
softmax(QK^T / sqrt(d_head)) V
```

#### With OT: `uot_cross_attention`

Add the normalized log transport plan to every head's logits:

```text
attention_logits = QK^T / sqrt(d_head)
                 + lambda_ot * log(P^T + minimum_mass)
```

The OT term has shape `[B, M, N]` and broadcasts across heads. Initialize learned
`lambda_ot=1.0`. Do not replace the Transformer attention distribution with `P`; the model
must be able to refine the OT prior.

Initial configuration:

```text
layers=1, heads=4, ffn_hidden=1024, dropout=0.2
```

### 6.5 Q-Former

Use a small fixed set of learned query tokens that cross-attend multimodal memory. This is
a lightweight Q-Former-style fusion module, not a claim of reproducing a particular
pretrained BLIP-2 checkpoint.

#### Without OT: `qformer`

Build multimodal memory by projecting and concatenating visual and question tokens:

```text
memory = [project(V), project(Q)]
queries -> self-attention -> cross-attention(memory) -> FFN
```

#### With OT: `uot_qformer`

Build grounded question tokens using the shared barycentric OT fusion, then use:

```text
memory = [project(V), H_ot]
```

The Q-Former architecture, query count, and parameterization remain unchanged. Because the
memory content differs, describe this as OT-augmented Q-Former rather than an attention-logit
prior ablation.

Initial configuration:

```text
query_tokens=8, layers=2, heads=4, ffn_hidden=512, dropout=0.2
```

The Q-Former output query sequence becomes decoder memory `[B, 8, d_model]`.

## 7. Configuration and CLI

Add serializable dataclasses:

```text
SANConfig
BANConfig
MUTANConfig
CrossAttentionFusionConfig
QFormerConfig
```

Each class validates dimensions, dropout, layers, ranks, heads, and query counts. Store
only the active method configuration in the checkpoint, plus `OTConfig` when OT is active.

Extend `--fusion` with all names in the experiment matrix. Add method options:

```text
--ban_glimpses 2
--ban_dim 256
--mutan_rank 5
--mutan_dim 256
--cross_fusion_layers 1
--qformer_queries 8
--qformer_layers 2
--fusion_dropout 0.2
```

Prefer a JSON `--fusion_profile` once the number of method fields grows. CLI flags may
override profile values, but the fully resolved configuration must be saved in checkpoints
and run metadata.

## 8. Diagnostics

Define common diagnostics where possible:

| Metric | Applicable methods |
| --- | --- |
| Memory length and norm | all |
| Attention entropy | all attention methods |
| Number of active glimpses | BAN |
| MUTAN factor/output norms | MUTAN |
| Learned OT-prior scale | BAN and cross-attention OT variants |
| Query attention entropy | Q-Former |
| OT cost, entropy, mass, residual, iterations, convergence | all OT variants |
| Prediction diversity and top-answer fraction | all |

Training code must use `fusion_spec.uses_ot` to decide whether OT diagnostics are expected.
Non-OT diagnostics should use a separate prefix such as `val_fusion_*` rather than
`val_ot_*`.

## 9. Checkpoint compatibility

Keep format version 3 if the payload schema remains unchanged and the complete architecture
is reconstructible from `model_config`. Bump the format only if payload semantics change.

Requirements:

- old version-2 SAN checkpoints continue to load;
- current version-3 OT and OT-SAN checkpoints continue to load strictly;
- every new fusion saves its normalized spec and method configuration;
- strict loading must reject missing or unexpected method parameters;
- resume is allowed only for the exact same fusion/configuration;
- benchmark runs always use new output directories and never auto-resume.

## 10. Tests

### 10.1 Shared contract tests

For every fusion name:

1. Construct the module from config.
2. Run variable-length batches with visual and question masks.
3. Check finite `[B, L, d_model]` memory and matching Boolean mask.
4. Perturb padded inputs and prove valid outputs are unchanged.
5. Backpropagate answer loss and check gradients reach the active fusion parameters.
6. Generate without reference answers.
7. Run online and cached-feature paths.
8. Save and strictly reload a checkpoint.
9. Verify diagnostic mode does not change evaluation logits.

### 10.2 Method-specific tests

- BAN: glimpse weights normalize over valid pairs; padded pairs have zero weight.
- UOT-BAN: setting `lambda_ot=0` reproduces non-prior BAN for identical inputs/weights.
- MUTAN: rank factors receive finite gradients; no full Tucker core is allocated.
- UOT-MUTAN: barycentric pooling is finite at minimum received mass.
- Cross-attention: query and memory lengths may differ; masks broadcast across heads.
- UOT cross-attention: OT prior shape is `[B, heads, M, N]` after broadcasting.
- Q-Former: output length equals configured query count regardless of input lengths.
- OT Q-Former: grounded question padding cannot affect query outputs.

### 10.3 Device tests

Run all unit and integration tests on CPU. Extend the opt-in MPS smoke test for one forward,
backward, and optimizer step per family. Run a CUDA mixed-precision smoke test when CUDA is
available. Sinkhorn stays in float32.

## 11. Automated benchmark

Use `scripts/run_fusion_benchmark.sh`. It creates this layout:

```text
results/fusion_benchmark/RUN_ID/
  san/none/seed_1105/
    command.txt
    train.log
    model/best.pt
    model/last.pt
    model/metrics.jsonl
  san/uot/seed_1105/
  ban/none/seed_1105/
  ban/uot/seed_1105/
  ...
  runs.csv
  runs.json
  aggregate.csv
  aggregate.json
  paired_deltas.csv
  paired_deltas.json
  paired_aggregate.csv
  paired_aggregate.json
```

The runner:

- refuses to overwrite an existing run directory;
- preflights all requested fusion names before training;
- uses a new model directory for every method/transport/seed;
- captures the exact command and combined stdout/stderr;
- stops on the first failed run;
- summarizes `metrics.jsonl` after all runs and calculates paired Balanced-OT-minus-no-OT,
  UOT-minus-no-OT, and UOT-minus-Balanced-OT deltas;
- does not evaluate the held-out test split by default.

Example:

```bash
chmod +x scripts/run_fusion_benchmark.sh
DEVICE=mps \
OT_PROFILE=configs/ot_mps.json \
SEEDS="1105 1106 1107" \
scripts/run_fusion_benchmark.sh
```

Quick smoke benchmark:

```bash
DEVICE=cpu \
OT_PROFILE=configs/ot_cpu.json \
METHODS="san mutan" \
SEEDS="1105" \
EPOCHS=2 \
RUN_ROOT=results/fusion_smoke \
scripts/run_fusion_benchmark.sh
```

To run only one or more selected pairs:

```bash
METHODS="san ban" SEEDS="1105" scripts/run_fusion_benchmark.sh
```

## 12. Statistical comparison

Use at least three paired seeds. For each family report:

```text
delta_F1(seed) = best_validation_F1(UOT, seed)
               - best_validation_F1(non-OT, seed)
```

Report:

- each seed's best epoch, validation loss, EM, and F1;
- mean and standard deviation across seeds;
- mean paired Balanced-OT-minus-no-OT, UOT-minus-no-OT, and
  UOT-minus-Balanced-OT deltas;
- train/validation F1 gap at the selected epoch;
- parameter count and training time;
- prediction diversity;
- image-shuffle and question-shuffle sensitivity;
- OT convergence and mass diagnostics for OT variants.

With a 100-example validation split, a `0.01` change may be one example. Do not declare a
winner from a single seed or a one-example difference.

After architecture and hyperparameter choices are locked using validation only, evaluate
one selected checkpoint per method/transport/seed on the held-out test split.

## 13. Milestones

### M0 — Benchmark infrastructure

- Add normalized fusion parsing and `uses_ot`.
- Add the runner and summarizer.
- Verify dry/preflight behavior and isolated result directories.

### M1 — Shared fusion interface

- Add `FusionInput`, `FusionOutput`, registry, and configuration serialization.
- Adapt legacy SAN and current OT-SAN without changing their checkpoint behavior.

### M2 — BAN pair

- Implement native BAN and OT-prior BAN.
- Complete CPU, cache, checkpoint, and device tests.

### M3 — MUTAN pair

- Implement low-rank MUTAN and OT-barycentric MUTAN.
- Complete the same test matrix.

### M4 — Cross-attention pair

- Implement native and OT-biased cross-attention.
- Verify attention mask and plan-orientation correctness.

### M5 — Q-Former pair

- Implement lightweight learned-query fusion and OT-grounded memory.
- Verify fixed output length and checkpoint reconstruction.

### M6 — Smoke benchmark

- Run two epochs, one seed, and all fifteen variants.
- Confirm artifacts, metrics, memory use, and failure handling.

### M7 — Controlled benchmark

- Run all fifteen variants with at least three paired seeds.
- Produce run-level and aggregate summaries.
- Run grounding diagnostics on each selected `best.pt`.

### M8 — Locked test evaluation

- Select configurations without inspecting test performance.
- Evaluate the locked checkpoints on test.
- Document limitations and avoid unsupported accuracy claims.

## 14. Acceptance criteria

Engineering is complete when:

1. All fusion names construct, train, generate, save, load, and resume correctly.
2. Every method returns valid masked decoder memory through online and cache paths.
3. OT and non-OT classification uses normalized `FusionSpec`, not string heuristics.
4. All CPU tests pass and target-device smoke tests are finite.
5. The benchmark runner produces isolated checkpoints, logs, and summary files.
6. Legacy checkpoints retain their existing behavior.

The comparison is complete when:

1. At least three paired seeds finish for all fifteen variants.
2. The aggregate report includes performance, generalization, grounding, cost, and
   parameter-count measures.
3. Test data is used only after validation choices are locked.
4. Conclusions distinguish explicit OT benefits from architecture-capacity differences.
