---
name: ot-vqa-testing
description: Choose and run focused checks for OT-VQA Python changes without deleting tests, datasets, checkpoints, or research outputs. Use when validating a code change.
---

# OT-VQA testing and validation

First inspect the repository's existing test files and available dependencies. The repository's `test.py` is the VQA evaluation program, not a unit-test runner. Do not delete `tests/`, `test.py`, caches, datasets, checkpoints, or reports as post-run cleanup.

For a focused change, run the relevant existing tests if present. For a solver or fusion change, cover shapes, masks/padded tokens, marginals and dustbin behavior, finite outputs, gradients, and supported devices when available. For CLI or data changes, use `--help` and small fixtures when available. Avoid launching full training or held-out test evaluation unless requested.

Report each command run and its result. If the repository has no relevant tests, say so and use an appropriate non-destructive check; do not create a new test framework unless the task calls for it.
