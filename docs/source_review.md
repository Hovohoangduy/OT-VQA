# Simplified Source Review

## Result

The active source now has one deployable model and one optional research teacher:

- deployable model: native Cross-Attention VQA;
- research component: training-only contrastive UOT alignment and attention distillation.

This removes multiple public branches that previously complicated construction,
checkpoint loading, diagnostics, tests, and experiment interpretation.

## Removed source

- `model/sans.py` and `model/ot_san.py`;
- BAN, MUTAN, aligned Cross-Attention, Q-Former, registries, and transport priors from
  `model/fusion_methods.py`;
- runtime projections, learned cost, learned marginals, barycentric fusion, and
  `TransportOutput` from `model/optimal_transport.py`;
- all runtime OT JSON profiles;
- transport-map visualization;
- fusion benchmark runner and aggregator;
- tests dedicated to removed architectures;
- old runtime-OT and fusion-family design documents.

## Retained source

- frozen ViT/DeiT patch extraction;
- frozen English BERT token extraction;
- native multi-head Cross-Attention;
- causal autoregressive answer decoder;
- raw and cached feature paths;
- generated EM/F1 checkpoint selection;
- output-diversity and modality-shuffle diagnostics;
- training-only UOT teacher, gate, fallback, and KL distillation;
- single-process and DDP execution;
- version-3 student and version-4 staged checkpoints.

## Public contract

`--fusion` accepts only `cross_attention`. The flag remains to make experiment records
explicit. Removed fusion names fail during CLI parsing. Removed and pre-cleanup
checkpoints fail during loading with a retraining message.

New checkpoints contain:

```text
architecture = cross_attention_only_v1
```

This marker prevents a legacy state dictionary from being interpreted as the smaller
model.

## Corrections preserved

The cleanup retains the previously corrected VQA contract:

- shifted answer inputs/targets;
- causal answer self-attention;
- memory padding masks;
- generation from BOS without reference answers;
- EOS termination and per-row padding;
- correct attention head merge order;
- ViT prefix-token removal without reshaping token dimensions;
- no double pixel rescaling;
- frozen encoders remain in evaluation mode;
- partial batches are processed;
- generated validation F1 selects checkpoints;
- feature-cache fingerprints are validated.

## OT interpretation

The supplied failed experiment did not distill OT. The gate rejected negative margin and
below-chance retrieval, after which KL and effective OT weight were zero. The later VQA
epochs therefore measured native Cross-Attention fallback.

The source deliberately keeps that safety behavior. It does not weaken the gate to make
an ineffective teacher appear successful.

## Verification

After cleanup:

```text
python -m compileall -q configs model utils train.py test.py predict.py tests
python -m unittest discover -s tests -q
```

Both commands pass. The focused suite currently runs 26 tests. These tests establish
software behavior and numerical validity, not VQA accuracy. Full paired training and
untouched-test evaluation remain required.
