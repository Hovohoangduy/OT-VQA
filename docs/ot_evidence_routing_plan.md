# Question-Conditioned Multi-Step OT Evidence Routing for General VQA

Status: implemented and unit-tested; VQA benchmark results are pending.

## 1. Objective and scope

Build a VQA model in which Optimal Transport (OT) is the core visual evidence
allocation mechanism at both training and inference. Support general visual
questions rather than assuming every question can be expressed through objects
and their relations.

The research hypothesis is that jointly allocating evidence to several
question-conditioned reasoning slots improves answer quality compared with
independent attention using the same features and reasoning architecture.

OT being structurally necessary does not establish that it improves accuracy.
Only matched experiments can establish its contribution. No performance gain is
claimed in this plan.

Initial implementation uses images, questions, and answer labels already supported
by the repository. Object annotations, scene graphs, question programs, and a
contrastive teacher are not required.

## 2. Evidence from previous experiments

- The supplied aligned-fusion comparison reported validation F1 of 0.383333 for
  non-OT and 0.381667 for UOT, with latency of 5.270873 and 14.807986 ms/example.
  This is approximately 2.81 times the latency for a small negative F1 difference.
  These are one-run results, not a statistically established general conclusion.
- The recorded teacher experiment failed its pre-distillation gate. Effective OT
  weight and distillation KL were zero. Its VQA result belongs to the native
  fallback, not a successfully distilled model.
- Possible causes include a mismatch between image-question retrieval and answer
  prediction, independently pretrained image/text representations, and inadequate
  evidence selection. These are hypotheses requiring ablation, not proven causes.

This proposal replaces word-to-patch matching with allocation from visual evidence
to learned information requests, optimized directly by answer supervision.

## 3. Architecture

```text
Image --> frozen visual encoder --> evidence tokens + positions + scale
                                                |
Question --> text encoder --> question context   |
                    |                           |
                    +--> initialize K slots     |
                                |               |
                    [question-conditioned cost matrix]
                                |
                    [semi-relaxed OT allocation] <--- evidence
                                |
                    [evidence readout + slot update]
                                |
                         repeat T steps
                                |
                    final grounded reasoning slots
                                |
Question context ----------> answer decoder --> answer
```

All visual input to the answer decoder must pass through the transport plan.
There is no parallel raw-image attention path. Question context remains available
to interpret the question and generate fluent answers.

Slots are latent information requests, not fixed object categories or individual
question words. Human-readable roles are possible but are not guaranteed.

## 4. Evidence bank

### First implementation

Use the existing frozen visual features and preprocessing:

- Spatial tokens retain their two-dimensional patch coordinates.
- Add a global summary token derived from those same spatial features.
- Add token-type embeddings distinguishing spatial and global evidence.
- Project evidence to the routing width with a trainable adapter.
- Add an explicit null token representing unavailable or unneeded evidence.

Give all matched routing controls exactly the same evidence tokens, including
global and null tokens. Adding a global token changes the representation and must
not be mistaken for an OT-specific gain.

The existing cache may already contain the necessary spatial features. Confirm
token ordering, removal of special tokens, patch geometry, preprocessing, and
feature dimensions before deriving positions. Do not infer a square grid without
checking the encoder metadata.

### Later extensions, only after the routing test

- Higher-resolution or multi-scale crops for small visual details.
- OCR features for reading-oriented tasks, with text, location, and confidence.
- A stronger visual or multimodal backbone.

Each extension needs its own matched attention comparison and cache version.
More evidence does not automatically mean better routing; overlapping scales can
duplicate evidence and distort coverage preferences.

The first version is a general architecture, not a claim of strong TextVQA,
counting, or world-knowledge performance with the present frozen backbone.

## 5. Question-conditioned reasoning slots

Let Q be the encoded question and q its masked pooled representation. Initialize
K slots from distinct learned slot embeddings and question context:

    S_i^(0) = LayerNorm(e_i + W_q q)

Slots may attend to valid question tokens during initialization and update. This
is text-only attention and does not bypass OT visual routing.

Start with K=4 and T=2 reasoning steps. Share routing and update weights across
steps to limit parameter growth. Use identical choices in matched controls.

For the first experiment, fix slot budgets a_i=1/K. Learned budgets can let a model
deactivate slots and obscure whether joint allocation helps. Test learned budgets
later using a positive floor and sum-to-one normalization.

## 6. Cost and visual preference

For each example and reasoning step, produce C with shape [K, N+1], including null.

    C_ij = f_theta(S_i, V_j, q, position_j, token_type_j)

Implement a lightweight compatibility function, such as normalized projected
query-key similarity plus bounded geometry/type terms. Use a fixed initial cost
scale and log its distribution. Avoid jointly unconstrained learned cost scale,
entropy strength, and marginal relaxation: their relative scales determine the
effective routing problem.

Predict a strictly positive visual preference b from the evidence and question,
normalized over valid tokens and null. For real tokens, mix learned preference
with a small uniform component over valid evidence to avoid irreversible exclusion.
Bound the null preference away from zero and one; its actual transported mass is
still determined by optimization.

Report a fixed visual-preference ablation. An unconstrained learned preference may
adapt to whatever allocation the model already wants and weaken OT coordination.

## 7. Semi-relaxed entropic OT

Use rows for slots and columns for evidence. For a single example:

    minimize over P >= 0, P 1 = a:
        <P, C>
        + epsilon * sum_ij P_ij (log(P_ij) - 1)
        + tau * KL(P^T 1 || b)

Use generalized KL, sum_j [r_j log(r_j/b_j) - r_j + b_j]. Valid marginals
a and b each sum to one. P therefore has unit total mass; the visual marginal
is softly constrained, but slot demand is hard-constrained.

This is semi-relaxed OT, not the current two-sided UOT teacher solver. Do not
silently reuse that solver with approximate large penalties for hard constraints.

Interpretation:

- Hard row budgets ensure each slot allocates its requested evidence mass.
- Soft visual preferences coordinate evidence use across slots.
- Unused regions are allowed, at a finite KL cost.
- Null absorbs unsupported requests without fabricating visual evidence.
- When tau approaches zero, the rows reduce to independent entropy-regularized
  softmax allocations. This provides an essential mathematical control.

Use log-domain scaling. With logK=-C/epsilon, alternate:

    log_u = log(a) - logsumexp(logK + log_v, evidence_axis)
    log_v = tau/(tau+epsilon)
            * [log(b) - logsumexp(logK + log_u, slot_axis)]

After the final visual update, recompute log_u so the returned plan satisfies row
budgets numerically. Run in float32 with autocast disabled inside the solver.
Mask invalid evidence exactly and guarantee at least the null token is valid.

Use fixed iteration counts during training for reproducible unrolled gradients
and to avoid per-iteration device-to-host synchronization. Start with 20
iterations. Test 10 only through an explicit accuracy/numerical/runtime comparison.

Report row error and a fixed-point residual. Deviation of the column sums from b
is expected under soft constraints and is not by itself non-convergence. Do not
claim convergence merely because final row normalization has small error.

## 8. Evidence readout and iterative reasoning

For slot i, compute the real-evidence contribution:

    E_i = sum_(j real) P_ij V_j / a_i
    null_fraction_i = P_i,null / a_i

Retain null fraction as a separate feature. Do not renormalize by real matched
mass, which would conceal uncertainty when most mass goes to null.

Read out spatial moments and token-type mass alongside appearance using the same
transport weights. Update slots with a residual gated MLP or recurrent update,
then allow slot-to-slot and slot-to-question interaction. Recompute costs for the
next step from the updated slots.

Repeated inspection is allowed. No universal penalty for revisiting the same
region: reading and comparisons may require it.

Weighted pooling can lose multiplicity and fine detail. Spatial summaries only
partly address this. If counting or reading fails, evaluate richer OT-mediated
readouts separately; do not claim counting ability from transport mass alone.

The final K slots become decoder memory. Preserve the existing answer tokenizer,
decoder capacity, loss masking, and generation policy for the first experiment.

## 9. Training objective and diagnostics

First objective:

    L = autoregressive answer cross-entropy

Backpropagate through transport iterations into costs, evidence projections,
question-conditioned preferences, and slot initialization/update. Frozen encoders
can continue using existing feature caches.

Do not enable the current contrastive teacher or attention distillation in these
runs. Add grounding or diversity losses only after diagnosing a specific failure,
as separately reported experiments.

Post-pilot revision: training diagnostics showed pairwise routing similarity near
0.98 and routing-query gradients orders of magnitude below the complete fusion
gradient. The implemented follow-up objective is therefore:

    L = autoregressive answer cross-entropy
        + lambda_div * mean_(i != j) cosine(query_i, query_j)^2

with `lambda_div=0.05` by default and a zero-weight ablation. Slot templates and the
shared question context are also normalized independently before being combined.
The default fixed Sinkhorn budget is increased from 20 to 40 because the pilot's
late-training fixed-point residual was about `0.005`, while the configured tolerance
is `0.001`. Existing checkpoints retain the iteration count saved in their model
configuration.
This is a targeted response to measured slot collapse, not evidence by itself that OT
improves VQA accuracy; matched multi-seed results remain required.

Required diagnostics, detached from the computation graph:

- Answer loss and generated EM/F1.
- Per-step normalized row entropy and pairwise slot-assignment similarity.
- Null fraction and visual marginal KL.
- Row residual, fixed-point residual, finite-plan rate, iteration count.
- Cost scale, gradient norms, and effective evidence coverage.
- Per-question-type results when reliable metadata exists.

Normalized row entropy must use P_i/a_i. Do not compare entropy of raw rows when
their budgets differ. Define coverage thresholds in experiment metadata.

## 10. Planned repository changes

| File | Planned responsibility |
| --- | --- |
| `model/ot_routing.py` (new) | Routing config, semi-relaxed solver, slot initialization, cost, readout, recurrent updates |
| `model/vqa_model.py` | Explicit routing-model construction and decoder-memory integration |
| `model/fusion_methods.py` | Existing native Cross-Attention baseline; shared controls only if appropriate |
| `configs/arg_parser.py` | Validated architecture/routing arguments |
| `train.py` | Direct-answer training, diagnostics, metadata, disallow incompatible teacher modes |
| `test.py`, `predict.py` | Load routing models and return optional diagnostics |
| `utils/checkpoint.py` | Explicit new architecture identifier and complete reconstruction metadata |
| `utils/feature_cache.py`, `precompute_features.py` | Check geometry metadata; version any changed cache schema |
| `tests/test_ot_routing.py` (new) | Solver, masking, gradient, invariance, model integration tests |
| `scripts/run_ot_routing_experiment.py` (new) | Small matched experiment runner, not the removed broad fusion grid |

The current checkpoint architecture is `cross_attention_only_v1`. Add a distinct
identifier such as `ot_evidence_routing_v1`; do not rewrite the identity of existing
Cross-Attention checkpoints. Preserve strict loading of currently supported
checkpoints and explicitly reject incompatible routing metadata.

These files and flags are now implemented. Accuracy and latency statements remain
experimental targets until the matched benchmark is run.

## 11. Starting configuration

```json
{
  "architecture": "ot_evidence_routing_v1",
  "slots": 4,
  "reasoning_steps": 2,
  "routing_dim": 256,
  "shared_step_weights": true,
  "slot_budget": "uniform",
  "visual_preference": "question_conditioned",
  "null_token": true,
  "epsilon": 0.1,
  "tau": 0.5,
  "sinkhorn_iterations": 20,
  "diagnostic_tolerance": 0.001,
  "teacher_distillation": false
}
```

Values are starting hypotheses, not tuned settings. Define exact cost scaling,
preference smoothing, null bounds, and solver tolerance conventions before the
first run and store them in checkpoints. Routing width may differ from decoder
width; use a learned output projection and match it across routing controls.

## 12. Staged implementation and experiments

### Stage A: solver and data verification

Implement and validate the solver independently. Verify cache/online feature
agreement and positions. Check split IDs, duplicate records, answer normalization,
and train-only vocabulary/statistics. No external training downloads are implied
by this planning document.

### Stage B: one-step matched prototype

Implement one-step slots with interchangeable softmax and OT routing. Use the same
initialization for shared weights, evidence, decoder, and optimizer. Run a small
overfit/smoke check to detect broken gradients and visual bypasses. This is not a
performance benchmark.

### Stage C: two-step pilot

Train these configurations with seed 42:

1. Current native Cross-Attention reference.
2. Slot reasoner with independent softmax routing.
3. Identical slot reasoner with semi-relaxed OT.
4. OT with tau=0, implementing the exact independent-routing limit.

The native baseline differs in architecture. The slot-softmax comparison isolates
the transport contribution. Match the softmax temperature to epsilon and row
budgets to a. Initialize the same shared tensors, even if model construction
consumes random numbers differently.

Run focused ablations for fixed versus learned visual preference and one versus
two steps only after the pilot is numerically sound. Keep tuning budgets balanced
between OT and softmax; do not compare tuned OT against an untuned control.

### Stage D: confirmation

For the selected design and matched controls, run seeds 1105, 1106, and 1107.
Use identical data, decoder, optimizer policy, early stopping, answer generation,
and generated-validation-F1 checkpoint selection. Evaluate the selected models on
the held-out test split after configuration selection is complete.

Report individual seed scores, paired differences, mean/std, and uncertainty.
Three seeds are a minimum practical check, not strong statistical proof.

## 13. Generalization and causal-use checks

- Report generated EM/F1 overall and for available question categories; also use
  the dataset's official scoring where available. Do not silently equate token F1
  with official VQA accuracy.
- Test a predefined held-out question family or composition where metadata allows.
- Treat cross-dataset transfer as a separate experiment with declared preprocessing
  and answer-scoring rules. Broad architectural scope alone is not generalization.
- Shuffle images among distinct image IDs, retaining questions, and measure loss
  of accuracy. Avoid same-image swaps.
- Perturb or permute valid transport assignments while preserving row mass and
  measure answer changes; distinguish intervention effects from natural accuracy.
- Remove high-evidence regions and compare against equal-sized random removal.
- Track question-only performance, output diversity, and majority-answer rate.

These checks establish visual dependence and evidence use; they do not alone
establish human-faithful explanations or causality in the real-world scene.

## 14. Efficiency on Kaggle two T4 GPUs

Support both single-GPU execution and one model trained with two-rank DDP. Two
independent runs, one per GPU, are a separate scheduling option and must not be
described as distributed training of one model.

Use cached frozen features, float32 transport, and mixed precision outside the
solver where supported. Record per-GPU batch size, world size, gradient
accumulation, effective batch size, and learning-rate policy.

Routing work scales approximately as O(T I K N), apart from feature projections
and slot updates. Measure backward memory; unrolled iterations retain gradients.
Small cost matrices do not guarantee low latency because kernel launches and
synchronization can dominate.

Measure single-GPU deployment latency with warm-up, GPU synchronization, fixed
batch size, identical generation limits, and the same hardware. Report separately:

- Routing module latency.
- Cached-feature model latency.
- End-to-end latency including encoders and any OCR/crops.
- Tokens generated, throughput, and peak memory.

Start with an engineering target of at most 2x native-baseline end-to-end latency.
This is a proposed budget, not a measured claim. If testing 10 instead of 20
iterations, require finite diagnostics and no more than 0.002 mean validation F1
loss before adopting it. Keep solver iteration choice separate from model quality.

## 15. Verification requirements

1. Nonnegative finite plans and near-exact row budgets on valid entries.
2. Exactly zero transport to padding; robust handling of null-only evidence.
3. Tau=0 agrees with row-softmax at the matching temperature.
4. Small problems agree with an independent constrained reference optimizer.
5. More iterations reduce fixed-point residual on controlled cases; column KL is
   not misreported as a convergence error.
6. Nonzero finite gradients reach cost, preference, evidence, and slot modules on
   nondegenerate fixtures; check selected gradients numerically in higher precision.
7. Joint permutation of evidence, positions, and masks preserves output in eval.
8. Padded question/evidence values cannot affect valid outputs.
9. No raw visual tokens bypass routing on their way to decoder memory.
10. Diagnostics on/off preserve eval outputs; cache/online paths agree within a
    documented dtype tolerance.
11. Strict save/reload preserves logits and deterministic generation; resume restores
    optimizer/RNG/config, and current native checkpoints still load.
12. CPU tests, available accelerator finite-forward/backward tests, and a Kaggle
    two-rank DDP smoke test with clean process-group shutdown.

## 16. Acceptance and stop criteria

Accept an OT-specific accuracy claim only if the paired mean improvement over the
matched slot-softmax model is positive across the three-seed experiment, the
held-out test mean also improves, and the effect is credible relative to run
variation. Report inconsistent seed signs and uncertainty rather than hiding them.

For deployment, additionally compare against native Cross-Attention and the
declared latency budget. A provisional practical target is at least +0.005 absolute
generated F1 over the stronger non-OT control; this is a project target, not an
expected outcome or statistical significance threshold.

Stop expansion and diagnose if:

- Slots collapse to indistinguishable assignments across most questions.
- Most allocation goes to null or a single global token without useful evidence use.
- OT matches the independent-routing limit and offers no measurable advantage.
- Shuffling images barely affects predictions.
- Numerical stability requires masking broken gradients or silently disabling OT.
- Gains disappear under matched features, parameter budgets, or repeated seeds.

Do not revive teacher distillation or add several new objectives to rescue an
unvalidated prototype. Record negative results and the exact failure mechanism.

## 17. Deliverables after implementation

- Minimal routing module, matched controls, and strict checkpoint integration.
- Reproducible single-GPU and two-T4 notebook commands, verified against real flags.
- Tests and saved numerical/efficiency diagnostics.
- Per-seed metrics and matched ablation tables.
- Updated README and HTML architecture documentation clearly separating measured
  results from proposed extensions.

Until those experiments are complete, the current native Cross-Attention model
remains the established baseline. This plan changes the research direction, not
the evidence about current VQA performance.
