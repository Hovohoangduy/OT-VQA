# OT Evidence Routing V2: Sparse, Detail-Preserving Transport for VQA

Status: the V2 architecture, controls, training options, checkpoints, interventions,
and focused tests are implemented. Benchmark stages remain pending; no V2 accuracy
gain is claimed.

Implemented components include routed patch memory, sparsemax/top-k preferences,
bounded learned or fixed cost scaling, question-conditioned keys, runtime tau
warm-up/ramp scheduling, the optional counterfactual loss, optimizer parameter
groups, gradient accumulation, stable gradient diagnostics, non-finite-step skipping,
and shuffled-plan reliance testing. The optional intermediate answer head remains
intentionally deferred until gradient measurements justify it.

## 1. Decision from the V1 run

The V1 solver is numerically healthy, but the learned transport is too diffuse and
the four-slot decoder interface discards too much visual detail. The next experiment
should therefore change the representation and learning signal, not increase the
Sinkhorn iteration count.

The supplied run reached its best generated validation F1 of `0.3763` at epoch 31.
By epoch 39, teacher-forced training F1 had increased to `0.5327`, while generated
validation F1 had fallen to `0.3465`. This is an optimization/generalization gap,
not a failure of the transport solver to converge.

At the best epoch:

| Signal | Observed value | Interpretation |
| --- | ---: | --- |
| Mean routing entropy | `4.924` | Effective support is about `exp(4.924) = 138` evidence tokens, so routing is close to dense averaging. |
| Cost standard deviation | `0.0543` | With `epsilon=0.1`, the cost contrast is only about `0.54 epsilon`; the entropic term dominates selection. |
| Query cosine similarity | `0.0061` | The query-orthogonality loss achieved its local objective. |
| Assignment similarity | `0.5757` | Orthogonal queries still produced substantially overlapping transport rows. |
| Coverage | `1.0` | Every patch receives nontrivial mass; this is coverage, not evidence selection. |
| Fixed-point residual | `5.7e-5` | Forty iterations are already more than sufficient at the configured tolerance. |
| Row residual | `7.1e-8` | The hard row marginal is satisfied. |
| Null fraction | `0.075` | Null routing is not the dominant failure. |

The repeated `fusion_gradient_norm=inf` values must be audited separately. The
current diagnostic squares all gradients in float32, which can itself overflow.
Before interpreting this as a model instability, compute the norm with a stable
float64 accumulator and log the fraction and maximum magnitude of non-finite
gradient elements. Optimizer steps with a genuinely non-finite norm must be skipped
and counted rather than silently applied.

This single run does not establish an OT effect. It must be compared with the
matched `softmax_evidence_routing` model from the same initialization and seed.
The native Cross-Attention result is a practical baseline, but it does not isolate
the contribution of OT because its memory representation is different.

## 2. V1 failure mechanism

V1 maps approximately 196 spatial patches plus global and null evidence into only
four final slot vectors. Each slot receives a barycentric average of nearly the
entire image. Fine attributes, small objects, spatial relations, and multiplicity
can disappear even when the transport plan is mathematically correct.

The normalized visual preference also asks the combined column mass to resemble a
distribution over every patch. With a high-entropy preference and `tau=0.5`, OT
encourages broad coverage. That behavior conflicts with VQA questions that often
need a small number of regions.

The query-diversity penalty is not an adequate correction. It makes projected slot
queries orthogonal, but does not ensure that their costs have useful contrast or
that their transport rows select distinct, answer-relevant evidence.

Finally, all trainable fusion and decoder layers are learned from scratch with the
same default learning rate of `1e-5`. For this 5,000-example subset, the router has
a weak and delayed answer-loss signal through two recurrent steps, unrolled OT, and
an autoregressive decoder. A matched optimization study is required before
attributing the score entirely to the routing formulation.

## 3. V2 architecture

### 3.1 Keep OT as the only visual path

Retain the question-conditioned plan `P` with shape `[K, N+1]`. Raw visual tokens
must never be appended directly to decoder memory. All visual content must be
modulated by `P`.

For each real evidence token `j`, derive:

```text
column_mass_j = sum_i P_ij
role_j = sum_i P_ij S_i / (column_mass_j + delta)
gate_j = N_valid * column_mass_j
```

Construct a routed patch token with no untransported residual:

```text
Z_j = gate_j * W_out(
          LayerNorm(V_j)
          + FiLM(LayerNorm(V_j), role_j)
      )
```

The factor `N_valid` keeps the mean gate near one and avoids shrinking every token
merely because total transport mass is one. Bound the gate only for numerical
stability and report the clipped fraction.

The decoder memory becomes:

```text
[projected valid question tokens ; final reasoning slots ; routed patch tokens]
```

Question tokens are an allowed nonvisual path. Routed patch tokens retain local
appearance and position, while the slots retain compact relational summaries. This
removes the four-vector visual bottleneck without creating a raw-image bypass.

### 3.2 Make evidence preference selective

Replace the dense softmax preference with a question-conditioned sparse preference.
Two implementations should be tested in this order:

1. `entmax-1.5` or sparsemax over real-evidence relevance logits, mixed with a very
   small uniform floor only for numerical stability.
2. A fixed top-`M` preference ablation with `M` in `{16, 32}`.

Log preference entropy, column-mass entropy, effective support, top-`M` transported
mass, and global-token mass. `coverage=1` should no longer be treated as a positive
outcome by itself.

### 3.3 Calibrate cost contrast

Use a bounded positive cost scale instead of relying on raw cosine spread:

```text
C_ij = -alpha * cosine(W_s S_i, W_v V_j)
alpha = alpha_min + (alpha_max - alpha_min) * sigmoid(raw_alpha)
```

Initialize `alpha` so that `std(C) / epsilon` is approximately `2` to `4`, compared
with about `0.54` at the V1 best epoch. Use one scale per reasoning step, bounded
for identifiability. Log the scale and ratio; do not allow `alpha`, `epsilon`, and
`tau` to drift without bounds simultaneously.

The evidence key should also be question-conditioned, for example with a bounded
FiLM transform from the pooled question. This lets the same patch expose different
features for color, counting, relation, and object questions.

### 3.4 Introduce coupling gradually

Train the matched independent router first, then introduce column coupling:

```text
epochs 1..E_warm: tau = 0
next E_ramp epochs: tau increases linearly to tau_target
remaining epochs: tau = tau_target
```

Start the search with `tau_target` in `{0.05, 0.1, 0.2}` rather than `0.5`. This
allows answer-relevant costs and the preference network to form before OT begins
coordinating the rows. Keep `epsilon` and every other setting identical in the
matched softmax comparison.

Forty Sinkhorn iterations are unnecessary for the observed residual. After V2 is
stable, compare 10, 20, and 40 iterations and select the cheapest count whose mean
F1 is within `0.002` of the 40-iteration reference and whose plans remain finite.

## 4. Learning objectives

Use autoregressive answer cross-entropy as the primary objective. Add only losses
that address a measured failure.

### 4.1 Replace query orthogonality with plan-level diagnostics first

Set the existing query-diversity weight to zero in the initial V2 pilot. It already
drove query similarity close to zero without preventing assignment overlap. Do not
add an assignment-repulsion loss until results are broken down by question type;
several slots may legitimately inspect the same object.

### 4.2 Counterfactual visual-reliance loss

On a fraction of each batch, pair a question with a different image whose answer
and image ID differ. Require the gold-answer sequence to score better with the true
image:

```text
L_cf = max(0, margin - log p(y | I, q) + log p(y | I_wrong, q))
L = L_answer + lambda_cf * L_cf
```

Warm `lambda_cf` from zero and keep it small. This provides a direct signal that the
OT-mediated visual path must matter, without using object boxes, programs, or an
external alignment teacher. Report results with `lambda_cf=0` as an ablation.

### 4.3 Optional intermediate answer supervision

If cost gradients remain much smaller than decoder gradients, attach a shared
training-only answer head to the routed state after each reasoning step. Its loss
must use the same gold answer vocabulary and be removed at inference. Add this only
after measuring V2 gradients; do not combine it with several new losses in the
first pilot.

## 5. Optimization protocol

The random router and decoder should not inherit a learning rate intended for a
pretrained backbone. With the image and question encoders frozen, begin a matched
search over:

- Router and fusion parameters: `1e-4` and `3e-4`.
- Decoder/output parameters: `1e-4` and `3e-4`.
- Five-percent linear warm-up followed by cosine decay.
- Effective batch size at least 32 through DDP or gradient accumulation.
- Weight decay `0.01` and gradient clipping at `1.0`.

Apply the same search budget to OT and its matched softmax control. Select the
training recipe on the pilot seed without examining the held-out test split.

Because validation loss continued to improve after generated F1 peaked, retain
generated-F1 checkpoint selection. Also log greedy token accuracy and the gold
answer log probability so generation errors can be separated from representation
errors.

## 6. Required experiments

### Stage A: trust the measurements

1. Replace the gradient diagnostic with stable accumulation and record non-finite
   elements, pre-clip norm, post-clip norm, and skipped steps.
2. Run a 128-example overfit test. Require at least `0.95` generated training F1.
3. Evaluate question-only, shuffled-image, and shuffled-plan interventions on the
   V1 best checkpoint.
4. Run the matched V1 softmax model from the same initialization and seed.

If the model cannot overfit 128 examples or image shuffling barely changes the
answer score, stop and fix that failure before implementing V2.

### Stage B: isolate the representation change

Compare from identical initializations:

1. V1 softmax with four-slot memory.
2. V1 OT with four-slot memory.
3. V2 softmax with routed patch memory.
4. V2 OT with routed patch memory, initially using the dense preference.

This stage answers whether lost patch detail, rather than OT itself, is the primary
bottleneck.

### Stage C: isolate the OT changes

Starting from the best V2 representation, test one factor at a time:

1. Dense versus sparse visual preference.
2. Fixed versus bounded learned cost scale.
3. Constant versus warm-started `tau`.
4. No counterfactual loss versus the selected `lambda_cf`.

Do not compare a tuned OT model with an untuned softmax control. The control receives
the same routed-patch memory, optimizer search, and auxiliary loss where applicable.

### Stage D: confirmation

Freeze the configuration, then run paired seeds `1105`, `1106`, and `1107` for:

1. Native Cross-Attention.
2. V2 matched softmax.
3. V2 OT.

Report every seed, paired differences, mean, standard deviation, latency, memory,
and intervention results. Evaluate the held-out test set only after freezing the
configuration.

## 7. Acceptance and stop criteria

An OT-specific success requires all of the following:

- Positive mean paired V2 OT-minus-softmax generated F1 across the three seeds.
- A provisional improvement of at least `+0.005` absolute F1 over matched softmax.
- No seed hidden because its difference is negative.
- A clear accuracy drop or gold-answer log-probability drop under image shuffling.
- Finite gradients and fewer than one skipped optimizer step per 1,000 updates.
- Deployment latency within the declared budget relative to Cross-Attention.

Stop adding OT complexity if routed-patch softmax improves substantially but OT does
not beat it. That result would show that detail preservation helped while column
coupling did not. Also stop if apparent gains vanish with identical initialization,
matched tuning, or repeated seeds.

## 8. Recommended immediate next run

Before implementing V2, run the Stage A checks and the existing matched softmax
control. The supplied OT log alone cannot distinguish among three possibilities:

1. OT coupling is harmful.
2. The shared four-slot bottleneck is harmful while OT is neutral.
3. The training recipe is under-optimizing both routing models.

That distinction determines whether V2 should focus on transport, representation,
or optimization and prevents another long run from answering the wrong question.
