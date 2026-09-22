# Contrastive Optimal-Transport Alignment and Attention Distillation for VQA

> This design was specified before implementation. Its Stage-1 contrastive teacher and
> Stage-2 attention-distillation path are now implemented. Accuracy claims still require
> the planned three-seed benchmark; the architecture HTML must not be updated with final
> claims until those measurements exist.

## Implementation status

The following components are implemented:

- training-only `OTContrastiveAligner` with 128-dimensional visual/text adapters;
- cosine UOT, uniform masked marginals, and float32 Sinkhorn;
- mass-normalized matching scores and symmetric hard-negative OT InfoNCE;
- detached FIFO negative queue with exact checkpoint restoration;
- final-layer Cross-Attention weight exposure only when diagnostics/training requests it;
- masked `KL(stopgrad(P) || A)` attention distillation;
- five-epoch teacher warm-up, validation decision gate, and scheduled student distillation;
- version-4 resumable `last_training.pt` and version-3 student-only `best.pt` export;
- shared student-initialization save/load controls for paired experiments;
- CPU unit and end-to-end tests for gradients, masks, collapse, queue, checkpoints, and
  deployment loading.

Stage 3 remains deliberately conditional: joint refinement should be added or run only
after Stage 2 demonstrates a mean validation improvement. The E0–E4 three-seed benchmark
and untouched test evaluation are experimental work, not implementation claims.

## 1. Motivation and current evidence

The current experiments tested OT as part of the VQA fusion computation. The strongest
model remains native Cross-Attention without OT:

| Configuration | Best validation F1 | Validation loss | Train–validation F1 gap | Latency / example |
| --- | ---: | ---: | ---: | ---: |
| Cross-Attention, no OT | **0.4110** | 1.5265 | 0.2270 | 5.29 ms |
| Cross-Attention, direct UOT prior | 0.3753 | 2.1971 | 0.4349 | 21.07 ms |
| Gated aligned Cross-Attention, no OT | 0.3833 | 1.4566 | 0.1515 | 5.27 ms |
| Gated aligned Cross-Attention, UOT | 0.3817 | 1.9430 | 0.4476 | 14.81 ms |

The gated alignment reduces the damage caused by adding the transport plan directly to
attention logits, but it still does not improve F1. It adds approximately 0.74 million
trainable parameters, increases latency by 2.81 times, increases validation loss by
0.4864, and increases the train–validation gap by 0.2961 relative to its paired non-OT
variant.

These results suggest four problems with the present use of OT:

1. OT is trained only through the answer-generation objective; no objective verifies that
   matched image–question pairs align better than mismatched pairs.
2. Frozen BERT and ViT were pretrained independently, so their token spaces are not
   inherently cross-modally comparable.
3. Runtime OT constrains or alters the strongest fusion path and adds substantial latency.
4. The small 1,000-example split allows OT cost, marginal, and fusion networks to memorize
   training correlations.

The proposed solution changes the role of OT. OT becomes a **training-time teacher for
representation alignment and attention supervision**. Native Cross-Attention remains the
student and the deployment architecture.

## 2. Research hypothesis

The testable hypothesis is:

> A UOT teacher trained to distinguish matched from mismatched image–question pairs can
> provide useful grounding targets for native Cross-Attention. Distilling those targets
> during training can improve VQA generalization while preserving non-OT inference cost.

This hypothesis has two required parts:

- **Alignment validity:** the OT teacher must score real image–question pairs better than
  hard negative pairs.
- **Task utility:** supervising Cross-Attention with the validated transport plan must
  improve held-out generated VQA F1.

If the first condition fails, attention distillation must not be attempted. If the first
condition succeeds but the second fails, OT may improve grounding without improving VQA
performance; that result must be reported honestly.

## 3. Proposed system

### 3.1 Training graph

```text
Frozen ViT patch tokens V ──→ visual alignment adapter ──┐
                                                        ├─→ UOT teacher P
Frozen BERT tokens Q ───────→ text alignment adapter ───┘       │
                                                                ├─ OT contrastive loss
                                                                └─ attention distillation target

Frozen ViT patches V ───────────────────────────────────────────┐
                                                               ├─ native Cross-Attention ─→ decoder
Frozen BERT question tokens Q ──────────────────────────────────┘           │
                                                                            └─ VQA loss
```

### 3.2 Inference graph

```text
Image → frozen ViT ───────────┐
                              ├─ native Cross-Attention ─→ autoregressive decoder
Question → frozen BERT ───────┘
```

The OT teacher, alignment adapters, negative-pair scoring, and Sinkhorn solver are not
executed during deployment inference.

## 4. OT alignment teacher

### 4.1 Shared alignment space

Let the frozen encoders produce image-patch tokens

\[
V \in \mathbb{R}^{B\times N\times D_v}
\]

and question tokens

\[
Q \in \mathbb{R}^{B\times M\times D_q}.
\]

Two small trainable adapters map them into a shared 128-dimensional space:

\[
Z_v = \operatorname{normalize}(f_v(V)), \qquad
Z_q = \operatorname{normalize}(f_q(Q)).
\]

Each adapter uses:

```text
Linear(input_dim, 128) → GELU → LayerNorm(128) → L2 normalization
```

The initial pairwise ground cost is cosine distance only:

\[
C_{ij}=1-Z_{v,i}^{\top}Z_{q,j}.
\]

A learned pairwise-cost MLP is explicitly excluded from the first implementation because
the existing learned OT configuration shows severe overfitting. It may be tested later as
an ablation only after the cosine teacher succeeds.

### 4.2 Masks and marginals

The question transport mask excludes:

- padding;
- `[CLS]` and `[SEP]` boundary tokens;
- any tokenizer-specific special tokens.

Image padding remains false for the fixed ViT patch grid. The first implementation uses
uniform marginals over valid positions. Learned question-conditioned marginals are excluded
from the first experiment to reduce capacity and isolate the alignment objective.

### 4.3 UOT configuration

The initial teacher configuration is:

```json
{
  "transport_type": "unbalanced",
  "marginal_mode": "uniform",
  "cost_type": "cosine",
  "ot_dim": 128,
  "epsilon": 0.1,
  "tau_visual": 0.5,
  "tau_question": 0.5,
  "max_iterations": 20,
  "tolerance": 0.001,
  "minimum_mass": 1e-8
}
```

Sinkhorn remains in float32. Adapter and student computations may use mixed precision.

### 4.4 Mass-normalized OT similarity

For image `I` and question `Q`, compute:

\[
P^{I,Q}=\operatorname{UOT}(C^{I,Q},a,b).
\]

The matching score is negative mass-normalized transport cost:

\[
s(I,Q)=
-\frac{\sum_{ij}P^{I,Q}_{ij}C^{I,Q}_{ij}}
{\sum_{ij}P^{I,Q}_{ij}+\epsilon}.
\]

The mass normalization is required. Without it, UOT could obtain an artificially favorable
score by transporting almost no mass.

## 5. Contrastive OT objective

For each image `I_i`, compare its true question `Q_i` against incorrect questions. The
image-to-question loss is:

\[
L_{I\rightarrow Q}
=-
\log
\frac{\exp(s(I_i,Q_i)/T)}
{\sum_k\exp(s(I_i,Q_k)/T)}.
\]

The symmetric question-to-image loss is:

\[
L_{Q\rightarrow I}
=-
\log
\frac{\exp(s(I_i,Q_i)/T)}
{\sum_k\exp(s(I_k,Q_i)/T)}.
\]

The teacher objective is:

\[
L_{OT\text{-}NCE}=
\frac{L_{I\rightarrow Q}+L_{Q\rightarrow I}}{2}.
\]

The initial temperature is `T=0.07`.

### 5.1 Hard-negative selection

Computing Sinkhorn for all `B²` image–question pairs is unnecessary. For every positive
pair:

1. Compute inexpensive similarities between masked-mean adapter embeddings.
2. Select the three highest-scoring incorrect pairs.
3. Run UOT for the positive pair and those three hard negatives only.

Prefer negatives with the same answer, similar question wording, or the same object with a
different attribute. These negatives reduce reliance on language and answer-frequency
shortcuts.

The intended effective batch size is at least eight. If device memory cannot hold eight
cached-feature examples, use batches of four and maintain a detached FIFO queue of 32
candidate negative embeddings.

## 6. OT-to-attention distillation

Native Cross-Attention produces attention probabilities

\[
A\in\mathbb{R}^{B\times H\times M\times N}.
\]

Transpose and normalize the positive transport plan across image patches for every valid
question token:

\[
\hat P_{ji}=
\frac{P_{ij}}{\sum_iP_{ij}+\epsilon}.
\]

Initially average student heads and the final fusion layer:

\[
\bar A_{ji}=\frac{1}{H}\sum_hA_{hji}.
\]

The distillation loss is:

\[
L_{distill}
=
\frac{1}{|\mathcal V|}
\sum_{j\in\mathcal V}
KL\left(
\operatorname{stopgrad}(\hat P_j)
\parallel
\bar A_j
\right),
\]

where `V` is the set of valid question tokens. `stopgrad` is mandatory: the teacher must
not move merely to imitate the student.

This differs from the unsuccessful direct-prior method:

```text
Rejected runtime design:
attention_logits = native_logits + lambda * log(P)

Proposed training design:
loss = VQA_loss + small_weight * KL(stopgrad(P), native_attention)
```

The student may disagree with an unreliable OT plan because distillation is a soft
auxiliary loss rather than a compulsory attention bias.

## 7. Complete training objective

The final objective is:

\[
L_{total}
=L_{VQA}
+\lambda_{NCE}L_{OT\text{-}NCE}
+\lambda_{distill}L_{distill}.
\]

Initial values:

| Hyperparameter | Initial value |
| --- | ---: |
| Contrastive temperature | 0.07 |
| `lambda_nce` | 0.05 |
| `lambda_distill` | 0.02 |
| Alignment-adapter learning rate | 0.0001 |
| OT dimension | 128 |
| Sinkhorn iterations | 20 |
| UOT visual/question `tau` | 0.5 / 0.5 |

Raw transport cost must not be added directly to the total loss. Minimizing positive-pair
cost without negatives permits representation collapse.

## 8. Training stages and decision gates

### Stage 0 — controlled baseline

1. Generate one common initialization checkpoint for each seed: `1105`, `1106`, `1107`.
2. Store Cross-Attention, decoder, projections, answer embeddings, and output-head weights.
3. Load those exact shared weights into the baseline and every experimental student.
4. Train native Cross-Attention for all three seeds.
5. Record generated F1, loss, latency, and train–validation gap.

This stage fixes the current fairness issue in which constructing additional OT modules
changes the random-number sequence and therefore changes downstream initialization.

### Stage 1 — alignment warm-up

For five epochs:

- freeze ViT, BERT, Cross-Attention, decoder, and answer head;
- train only the visual/text adapters;
- optimize the unscaled `L_OT-NCE` with the dedicated alignment-adapter learning rate;
- evaluate positive/negative score margin and bidirectional retrieval accuracy.

Proceed to distillation only when:

- positive pairs score better than negatives on validation data;
- retrieval accuracy is above random chance;
- embeddings have nonzero per-dimension variance;
- transport plans, mass, entropy, and residuals are finite;
- matched mass does not collapse toward zero.

If these checks fail, stop. Do not build VQA training around an invalid teacher.

### Stage 2 — frozen-teacher distillation

- freeze the validated OT teacher;
- train native Cross-Attention and the answer decoder;
- optimize `L_VQA + lambda_distill * L_distill`;
- linearly increase `lambda_distill` from 0 to 0.02 over five epochs;
- use generated validation F1 for checkpoint selection and early stopping.

### Stage 3 — optional joint refinement

Run only if Stage 2 improves mean validation F1:

- continue `L_OT-NCE` with weight 0.01;
- keep the OT plan detached in `L_distill`;
- optionally unfreeze the final ViT and BERT layer;
- use a pretrained-encoder learning rate ten times smaller than the student learning rate.

## 9. Low-data strategy

The included engineering split has only 1,000 training examples. This is sufficient for
smoke tests but may be insufficient for learning a reliable cross-modal space.

The preferred final experiment uses the full available GQA training split. If training is
restricted to the 1,000-example subset, initialize or supervise the alignment adapters
with a pretrained vision-language model such as CLIP. A frozen CLIP teacher can provide
patch/text similarity targets, but tokenization differences must be mapped through word
offsets before comparing its transport plan with BERT-token attention.

This CLIP teacher is a fallback, not part of the first implementation. The first decision
gate is whether contrastive OT over the current cached ViT/BERT features learns useful
retrieval on validation data.

## 10. Software design

### 10.1 New training-only components

Add an `OTContrastiveAligner` responsible for:

- visual and text adapters;
- masked cosine cost;
- UOT plans for explicit positive/negative pair indices;
- mass-normalized transport scores;
- transport diagnostics.

Add a loss module responsible for:

- symmetric OT InfoNCE;
- masked OT-to-attention KL divergence;
- collapse diagnostics.

The existing `VQAModel` remains the inference student. Cross-Attention must expose its
attention maps during training without changing ordinary evaluation output.

### 10.2 Configuration

Introduce a training configuration containing:

```text
alignment_mode = none | ot_contrastive_distill
alignment_warmup_epochs = 5
ot_negative_count = 3
ot_contrastive_temperature = 0.07
ot_contrastive_weight = 0.05
ot_distill_weight = 0.02
ot_distill_warmup_epochs = 5
```

Store these settings and teacher state in resumable training checkpoints. Export the
selected student as a normal inference checkpoint that does not require the teacher.

### 10.3 Checkpoints

Use two artifacts:

- `last_training.pt`: student, OT teacher, optimizer, scheduler, RNG, queue, and stage state;
- `best.pt`: deployment student only, loadable by the existing evaluation and prediction
  entrypoints without Sinkhorn.

Existing version-2 and version-3 checkpoints must remain loadable. If a new checkpoint
format is required for training-only state, use version 4 while preserving version-3
student export.

## 11. Required experiments

| Experiment | OT-NCE | OT distillation | Runtime OT | Purpose |
| --- | ---: | ---: | ---: | --- |
| E0 Native Cross-Attention | No | No | No | Primary baseline |
| E1 Alignment adapters only | No | No | No | Control for added projections |
| E2 Contrastive OT teacher | Yes | No | No | Test representation alignment |
| E3 OT-distilled Cross-Attention | Yes | Yes | No | Proposed model |
| E4 Runtime gated UOT | No | No | Yes | Existing negative reference |

Run every experiment with seeds `1105`, `1106`, and `1107` using identical shared
initialization, splits, optimization settings, checkpoint selection, and stopping rules.

### 11.1 Restricted hyperparameter search

Tune in sequence rather than as a full Cartesian grid:

1. Select teacher settings using retrieval metrics:
   - `tau`: 0.5, 1.0;
   - Sinkhorn iterations: 10, 20.
2. Fix the teacher.
3. Select `lambda_distill`: 0.01, 0.02.
4. Test `lambda_nce`: 0.01, 0.05 only during optional joint refinement.

No setting may be selected using the held-out test split.

## 12. Metrics and diagnostics

### 12.1 VQA metrics

- generated validation and test F1;
- exact match;
- teacher-forced token loss;
- train–validation F1 gap;
- output diversity;
- majority-answer fraction.

### 12.2 Alignment metrics

- positive-minus-hard-negative OT score margin;
- image-to-question retrieval accuracy;
- question-to-image retrieval accuracy;
- transport entropy and matched mass;
- Sinkhorn residual and convergence rate;
- adapter feature variance;
- mean OT-to-attention KL divergence.

### 12.3 Grounding tests

- image-shuffle F1 decrease;
- question-shuffle F1 decrease;
- patch-mass visualization;
- student attention visualization;
- agreement between OT and student attention.

### 12.4 Efficiency

- training-time OT overhead;
- deployment latency per example;
- peak inference memory;
- deployed student parameter count;
- assertion that the deployment forward path never calls Sinkhorn.

## 13. Tests required before experiments

1. Positive and selected-negative pair indices produce correctly shaped costs and plans.
2. Mass-normalized scores remain finite when matched mass is small.
3. Padded and special question tokens receive zero transport and distillation weight.
4. OT-NCE gradients reach both adapters.
5. The detached distillation target does not receive gradients from `L_distill`.
6. Distillation gradients reach Cross-Attention query/key parameters.
7. Identical positive and negative embeddings trigger the collapse diagnostic.
8. Hard-negative selection never selects the positive pair.
9. The negative queue is restored exactly after checkpoint resume.
10. Common initialization produces identical student weights across E0–E4 before
    training.
11. Enabling training-only OT does not alter ordinary inference logits when student
    weights are held fixed.
12. Exported `best.pt` loads through existing `test.py` and `predict.py` paths.
13. CPU and available CUDA/MPS forward and backward passes remain finite.

## 14. Acceptance criteria

The proposed model is accepted as a VQA-performance improvement only when:

1. Three-seed mean generated validation F1 exceeds native Cross-Attention by at least
   0.005.
2. Held-out test F1 also improves.
3. No paired seed decreases by more than 0.005.
4. Deployment latency remains within 10% of native Cross-Attention.
5. Positive OT pairs score consistently better than hard negatives.
6. Image shuffling causes a larger F1 reduction than for the baseline, indicating stronger
   visual dependence.
7. The exported inference checkpoint contains no active OT computation.

If alignment metrics improve but VQA F1 does not, the result must be described as improved
grounding rather than improved VQA performance. If the OT teacher cannot pass Stage 1,
implementation stops before distillation.

## 15. Implementation order

1. Reproduce the three-seed native Cross-Attention baseline with common initialization.
2. Implement the training-only alignment adapters and cosine UOT teacher.
3. Implement mass-normalized positive and hard-negative scoring.
4. Add OT-NCE, retrieval metrics, and collapse checks.
5. Complete the Stage-1 alignment warm-up experiment.
6. Expose Cross-Attention maps through a training-only output contract.
7. Implement detached OT-to-attention KL distillation.
8. Add staged training, checkpoint resume, and student-only export.
9. Run E0–E4 with three paired seeds.
10. Evaluate selected checkpoints on the untouched test split.
11. Update the architecture HTML and result tables only after measured results exist.

## 16. Expected outcome

The solution is designed to preserve what currently works—the native Cross-Attention VQA
path—while giving OT a measurable and falsifiable role. OT must first demonstrate valid
image–text matching, then act as a soft teacher, and finally disappear from deployment.
This removes the principal failure modes observed in runtime OT fusion: compulsory
alignment, excess inference latency, and high-capacity end-to-end overfitting.
