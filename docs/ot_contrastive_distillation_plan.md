# Historical Training-Only OT Alignment Experiment

This path remains available for reproducibility, but the active OT architecture is now
question-conditioned multi-step evidence routing. See
`docs/ot_evidence_routing_plan.md`. The failure analysis below explains why the older
teacher experiment is not evidence of an OT VQA gain.

## Status

The repository retains this native Cross-Attention plus optional training-only UOT
teacher path for reproduction. A separate runtime OT evidence-routing architecture now
exists; it does not use this teacher or its distillation objective.

The existing teacher is implemented and numerically stable, but the supplied experiment
did not validate it:

| Diagnostic at epoch 10 | Value | Required interpretation |
| --- | ---: | --- |
| Positive-hard-negative margin | -0.0421 | Must be positive; failed |
| I2Q retrieval | 0.246 | Four-candidate chance is 0.25; failed |
| Q2I retrieval | 0.219 | Four-candidate chance is 0.25; failed |
| VQA KL | 0.0000 | Teacher was not used |
| Effective OT weight | 0.0000 | Native fallback |

The resulting best validation F1 of `0.3323` belongs to native Cross-Attention fallback.
It cannot be used to evaluate OT.

## Current implementation

### Teacher inputs

- frozen ViT spatial patches `[B,N,Dv]`;
- frozen BERT content tokens `[B,M,Dq]`;
- Boolean visual/question padding masks.

### Alignment adapters

Each modality has an independent adapter:

```text
Linear(input_dim, ot_dim) → GELU → LayerNorm → L2 normalization
```

### Transport

The current teacher uses cosine cost, uniform masked marginals, UOT, and float32
log-domain Sinkhorn. Default numerical settings are epsilon `0.1`, tau `0.5`, 20
iterations, tolerance `1e-3`, and minimum mass `1e-8`.

### Contrastive objective

Pooled adapter similarity chooses hard negative candidates. Exact UOT scores are then
computed for the matched pair and selected mismatches. Symmetric InfoNCE trains I2Q and
Q2I ranking. A detached FIFO queue supports small batches.

### Gate and fallback

The teacher must demonstrate positive margin, above-chance bidirectional retrieval,
finite transport, and non-collapsed features. The recommended development policy is:

```text
--ot_gate_failure_policy error
```

`fallback` is useful only when a native completion is wanted. It disables distillation
and records `ot_gate_fallback=true`.

### Distillation

After a pass, the frozen positive plan supervises mean final-layer Cross-Attention using
masked KL divergence. `best.pt` exports only the student; `last_training.pt` preserves
teacher and queue state for resume.

## Root cause requiring redesign

The current teacher attempts to learn the shared ViT/BERT space and local transport at
the same time. These encoders were pretrained independently. On limited VQA data, shallow
random adapters do not reliably make true image-question pairs outrank hard negatives.

The current score also divides transport cost by matched mass:

```text
-sum(P*C) / sum(P)
```

This discards UOT coverage information. A mismatched pair can obtain a competitive score
by transporting little mass through coincidentally similar token pairs.

## Next implementation, not yet applied

The next change should be global-to-local alignment:

```text
Stage 1A: pooled image/question global InfoNCE
Stage 1B: local UOT objective
Stage 1C: retrieval and collapse gate
Stage 2: gated attention distillation
```

Recommended teacher loss:

```text
Lteacher = Lglobal_InfoNCE + alpha * Llocal_UOT
```

Recommended pair score uses complete regularized UOT energy:

```text
S(V,Q) = -[
  <P,C>
  + tau_v KL(P1 || a)
  + tau_q KL(Pᵀ1 || b)
  - epsilon H(P)
]
```

Implementation requirements:

1. Pretrain global adapters before enabling local transport.
2. Preserve matched-mass and marginal-deviation information in the score.
3. Filter false negatives that share an image or duplicate question identity.
4. All-gather candidates across DDP ranks.
5. Synchronize or replace per-rank negative queues.
6. Select the best teacher epoch by validation margin/retrieval instead of always using
   the final warm-up epoch.
7. Keep the existing gate strict.
8. Enable attention KL only after the redesigned teacher passes.

## Controlled evaluation

For each seed `1105`, `1106`, and `1107`:

1. Save one common student initialization.
2. Train native Cross-Attention.
3. Train the OT-distilled student from the identical initialization.
4. Select both by generated validation F1.
5. Evaluate selected checkpoints on an untouched test split.

Report generated EM/F1, validation/test loss, train-validation gap, latency, teacher
margin and retrieval, KL magnitude, effective OT weight, modality-shuffle degradation,
output diversity, and majority-answer rate.

Claim improvement only when the three-seed paired mean is positive and the untouched
test result also improves. Until then, native Cross-Attention remains the deployment
model and performance baseline.
