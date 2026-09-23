# Source Review: OT Evidence Routing

## Implemented result

The repository now contains a runtime OT architecture in which every visual token sent
to the answer decoder is allocated through semi-relaxed OT. It works with raw images and
feature caches, single-GPU training, two-rank CUDA DDP, evaluation, prediction, strict
checkpoints, and architecture-neutral diagnostics.

The implementation adds:

- `model/ot_routing.py` with a float32 hard-row/soft-column solver;
- general spatial, global, and null evidence tokens;
- question-conditioned multi-step reasoning slots;
- matched softmax and `tau=0` controls;
- routing CLI and architecture-aware checkpoints;
- a reproducible experiment runner;
- cache grid metadata and validation;
- focused solver, gradient, mask, integration, and checkpoint tests.

## Preserved behavior

Native Cross-Attention and its checkpoint marker remain supported. The answer decoder,
answer normalization, shifted targets, autoregressive generation, EOS handling, frozen
encoders, feature caches, generated-F1 selection, and existing DDP behavior remain shared.

The historical contrastive UOT teacher remains reproducible only with Cross-Attention.
Routing models reject that alignment mode because they train OT directly from the VQA
answer objective.

## Numerical contracts

- Slot row sums equal fixed uniform budgets.
- Padded evidence receives exactly zero returned mass.
- Log-domain padded columns retain finite gradients.
- Transport always executes in float32.
- `tau=0` equals the independent-softmax implementation.
- Null mass remains explicit in the readout.
- Grid dimensions must match spatial-token count.
- Diagnostics do not alter routing outputs.

## Checkpoint contract

```text
cross_attention            -> cross_attention_only_v1
ot_evidence_routing        -> ot_evidence_routing_v1
softmax_evidence_routing   -> ot_evidence_routing_v1
```

OT and softmax controls can exchange a common initialization when all model fields except
their routing mode match. Other mismatches fail strictly.

## Verification status

The full local test suite and compile check pass after implementation. These checks
establish numerical and software behavior. CUDA/MPS hardware checks, the Kaggle DDP
smoke test, multi-seed training, test-set accuracy, and latency measurement require the
target hardware and dataset and remain experiment work.

No VQA performance improvement is claimed from source implementation alone.
