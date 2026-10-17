---
name: performance-profiling
description: Measure OT-VQA training and inference runtime or memory, identify the bottleneck, and validate targeted optimizations. Use when throughput or resource use is the task.
---

# OT-VQA performance profiling

Measure before optimizing. Use a fixed checkpoint, device, batch, image size, question lengths, and warm-up policy. Separate data loading, image encoding, cost construction, Sinkhorn iterations, decoder work, and evaluation metrics when possible. Synchronize the accelerator around timed regions and report device, batch size, precision, and repetitions.

The pairwise OT cost and plan scale with `B × N × L`; higher image resolution increases spatial token count `N`. Read `model/optimal_transport.py` and the active checkpoint configuration before proposing changes. The current solver already uses log-domain updates and FP32 arithmetic; do not paste generic linear-domain Sinkhorn or custom backward examples over it.

For every optimization, compare both runtime/memory and validation quality with a controlled baseline. Treat `torch.compile`, implicit differentiation, lower-rank OT, reduced iteration counts, and mixed precision as research changes: verify numerical behavior, gradients, marginal residuals, checkpoint compatibility, and output quality before recommending them. Do not claim a speedup without a measured benchmark.
