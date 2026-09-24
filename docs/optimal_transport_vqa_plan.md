# Plan: optimal partial transport for GQA VQA

## Scope and evidence

This plan adapts the *matching mechanism* in Pan et al., [Learning Image-Text Matching with Optimal Partial Transport](https://arxiv.org/html/2603.14349v1), to this repository's answer-generation task. The paper evaluates image-caption retrieval on Flickr30K and MS-COCO using recall at K. It does **not** establish a gain in GQA answer accuracy. Its key elements are local image/text features, cosine cost, entropic Sinkhorn transport, global-feature dustbins for irrelevant fragments, and a retrieval ranking loss. The first four are useful here; the ranking loss requires a VQA-specific replacement.

This repository has no `data/` directory, GQA CSVs, trained checkpoint, or recorded baseline metrics in the current workspace. Accuracy and training speed cannot yet be measured. The source review below describes the current code, not measured model quality. `docs/source_review.md` documents earlier fixes, but its final paragraph about checkpoint resume is outdated: the current `train.py` does resume version-3 checkpoints.

## Current pipeline and gaps

| Area | Current source | Consequence for the proposed method |
| --- | --- | --- |
| Data | `utils/download_gqa.py` defaults to 1,000 train, 100 validation, and 100 test images; it selects one question per image. `utils/vqa_dataset.py` returns image, question, and one answer. | This is enough for a smoke test, but likely too small for a meaningful architecture comparison. Preserve image-disjoint splits and increase the sampled image counts if more training data is needed. |
| Vision | `model/features_extraction.py::ImageEmbedding` freezes ViT and returns all hidden states `[B,197,768]` for the default 224/16 model. The processor runs on CPU for every batch. | Patch tokens already exist, but the current fusion collapses them. Exclude the ViT CLS token from the OT patch set; retain it as optional global context. Preprocessing and feature caching need profiling. |
| Question | `QuestionEmbedding.encode_tokens()` exposes BERT token states and a mask for special/padding tokens. `forward()` instead compresses these states to one LSTM vector. | Use `encode_tokens()` for the OT path. Never transport mass to PAD, CLS, or SEP. Handle varying valid-token counts, including very short questions. |
| Fusion | `model/vqa_model.py::encode()` projects patches, runs SAN with one question vector, then returns one decoder-memory token. `model/sans.py` implements the SAN. | Replace SAN in a separately selectable OT model mode. The transport plan should directly determine grounded memory delivered to the answer decoder. |
| Decoder | `model/decoder_model.py` supports memory length greater than one and a memory mask. `VQAModel.decode()` currently passes no memory mask. | Feed one grounded memory token per valid question token plus a global token, and pass the memory padding mask. Keep causal answer masking. |
| Training | `train.py` optimizes teacher-forced answer cross-entropy and selects `best.pt` by validation loss. `test.py` generates answers but recomputes image/question encodings for teacher-forced loss. | Begin with the same answer loss, then test answer-aware auxiliary supervision. Cache/reuse encodings during evaluation if useful. Track all six generated metrics. |
| Checkpoints and CLI | `configs/arg_parser.py` and `utils/checkpoint.py` assume SAN; checkpoints store model configuration and training state. | Add a `fusion`/architecture field and OT hyperparameters, increment checkpoint format, and retain strict loading of old SAN checkpoints. Require explicit architecture selection on resume. |
| Metrics and diagnostics | `utils/metrics.py` reports case-insensitive EM, SQuAD-style Token-F1, BLEU-1/2, ROUGE-L, and BERTScore-F1. `diagnose_training.py` tests image/question reliance, prediction collapse, and answer distribution. | Add question-type slices, transport diagnostics, and speed/memory measurements. Re-run shuffled-image tests for OT. |
| Tests and inference | `tests/test_logic.py`, `tests/test_device.py`, and `predict.py` cover basic generation, data and device contracts. | Add OT unit tests, one tiny end-to-end learning test, checkpoint round trip, and single-image inference. |

The present ViT is frozen, while BERT and the decoder are trained. Answer embeddings are instantiated separately from the question BERT, adding memory use. Image resize forces every picture to 224 x 224, which may harm small-object and spatial questions. Treat these as measured baseline limitations and potential later improvements, not as benefits guaranteed by OT.

## Proposed OT fusion

```text
image -> frozen ViT -> patch tokens V [B,K,Dv] -> projection + normalization
question -> BERT -> valid wordpiece tokens T [B,L,Dt] -> projection + normalization
                                      V,T -> partial Sinkhorn plan P [B,K+1,L+1]
                                                   |
                                  local-local mass P[:K,:L] -> grounded question memory
                                                   |
                                causal answer decoder -> generated answer
```

1. **Fragments.** Set `V = ViT[:,1:,:]` and use only BERT tokens where the returned padding/special-token mask is false. Project both to `d_model` (384 initially), then apply L2 normalization **for the cost**. Preserve unnormalized projected values for the decoder. For default settings, `K=196` and `L<=26`.
2. **Cost.** Use `C_ij = 1 - cos(V_i,T_j)` on valid pairs. Append one image-global and one question-global dustbin embedding, following the paper. An image-global mean over valid patches and BERT CLS/question mean are initial choices. Explicitly specify and test dustbin-to-dustbin cost; otherwise it can consume dustbin capacity and force unwanted local matches. The implementation starts with cost `1.0` and exposes `--ot_dustbin_cost` for ablation.
3. **Partial mass.** Implement an extended balanced transport problem with configurable dustbin capacity. Start with equal valid-patch and valid-question marginals and 20% dustbin capacity on each side: each local patch receives `0.8/K`, each local token `0.8/L`, and each dustbin `0.2`. These are *VQA adaptation defaults*, not paper-reported optimum. Sweep 0, 0.1, 0.2, and 0.4; measure local-local matched mass and dustbin usage. Ensure the full row and column marginals both sum to one.
4. **Solver.** Implement batched log-domain Sinkhorn in FP32, including under mixed precision. Start with entropy coefficient `epsilon=0.05` and 20 iterations; sweep `0.02, 0.05, 0.1` and 10/20/40 iterations. Mask invalid rows/columns before normalization; avoid `log(0)` and rows whose every destination is masked. Test finite outputs and marginal residuals. The paper uses `lambda=0.02` for retrieval, which is a reference point, not a tuned GQA value.
5. **Grounded decoder memory.** For each valid question token `j`, compute `A_j = sum_i P_ij V_i / max(sum_i P_ij, tiny)` from the **local-local block only**. Use the fraction of its marginal that matched local patches as a gate: `g_j = sum_i P_ij / beta_j`. Form `M_j = LayerNorm(T_j + g_j * W A_j)`. Add one learned projection of global image/question context for questions whose answer depends on scene-wide information. The decoder sees `M` and its padding mask; SAN is absent in OT mode. This makes the transport plan the operative fusion step while retaining language context.
6. **Objective.** First train with the existing shifted answer-token cross-entropy. The paper's hardest-negative image-caption retrieval loss should not be copied verbatim: common GQA questions can correctly pair with many images, causing false negatives. If answer loss alone lets the model ignore vision, add a small answer-aware contrastive loss using image swaps only among examples with comparable questions and different answers. Verify it improves generated answers and shuffled-image sensitivity before keeping it.

The paper's equations have details to check against code: `1 - v·t` is half the squared Euclidean distance only for unit-normalized vectors, and its printed triplet hinge appears to put positive and negative scores in the opposite order from the stated ranking goal. Use a separately tested ranking formulation if an auxiliary loss is added. The reported retrieval scores must not be presented as VQA gains. The paper's appendix also notes difficulty with relational words and distinguishing entities, directly relevant to GQA.

## Implementation sequence

Stages 1 and 2 are implemented. The downloader keeps its original one-question-per-image format, with rate-limit handling and resumable page caching. Stages 3 to 5 have per-example predictions, runtime reporting, transport diagnostics, and paired comparison tooling. Real GQA training, hyperparameter selection, and final accuracy claims remain pending because this workspace contains no completed dataset or checkpoints.

| Stage | Changes | Exit check |
| --- | --- | --- |
| 0. Establish baseline | Prepare larger, image-disjoint GQA train/dev/test CSVs; record data counts, question/answer lengths and type mix. Run current SAN model with fixed seed and configuration. Save all six generated-answer metrics, per-type scores, latency, throughput, peak memory, and shuffled-image sensitivity. | Reproducible dataset manifest and at least one complete baseline run; three seeds for final comparison. |
| 1. OT core | Add `model/optimal_transport.py` with projections, masked cost, dustbins, log-Sinkhorn, and diagnostics. Unit test rectangular matrices, variable lengths, padding, marginal sums, gradients, and FP16/BF16 input with FP32 solve. | No NaN/Inf, small marginal residual, nonzero gradients, stable one-example behavior. |
| 2. VQA integration | Add an `ot` fusion choice to `VQAModel.encode`; preserve SAN as `san`. Pass grounded memory and memory mask through train/generate/evaluate. Update CLI and checkpoint schema. | Tiny-model overfit/generate test succeeds; SAN regression tests and old SAN loading still pass. |
| 3. Controlled training | Train SAN, full OT without dustbins, and partial OT with identical data, backbones, answer decoder, seeds, schedules, and model selection. Search OT settings on dev only. | Test split remains untouched until the model choice is frozen. Record runtime and memory alongside accuracy. |
| 4. Diagnose and refine | Compare count, color, relation, attribute and spatial questions; inspect top patch-token transport and dustbin mass; test same-question/different-image pairs and shuffled images. Consider answer-aware negative loss, modest ViT unfreezing, or resolution changes one at a time. | Keep each addition only if repeated dev gains exceed seed variation and image dependence does not regress. |
| 5. Final report | Evaluate the frozen selection once on the held-out test split. Report mean and standard deviation across seeds, paired bootstrap confidence intervals for all six metrics, parameter count, latency, throughput, and peak VRAM. | Claim improvement only if the selected metric's interval excludes zero and the compute cost is acceptable for the intended deployment. |

For PlantExpertVQA, report the six paper metrics on generated answers and use validation loss for checkpoint selection. For GQA, generated exact match remains a useful primary metric, with token F1 secondary. Report answer-type breakdown to expose gains hidden by the average. The lexical metrics are single-reference string metrics rather than an official multi-reference GQA scorer; record that distinction. A practical first target is a repeatable 2 percentage-point absolute EM gain over SAN with no collapse in image reliance. This is a decision threshold, **not** a prediction from the retrieval paper.

## Runtime priorities and risks

- **Likely highest impact:** More representative training data. The default downloader supplies one QA per image and only 1,000 train examples, which is a weak setting for judging a new fusion module.
- **Training throughput:** Move/avoid the current CPU `AutoImageProcessor` round trip in each `ImageEmbedding.forward()` or cache frozen ViT patch features keyed by image ID and encoder/preprocessing version. Cache is valid only while ViT is frozen and augmentations are fixed.
- **Memory:** Share or avoid duplicate BERT embedding initialization where possible; checkpoint and version the architecture change. Profile the OT cost matrix and decoder memory at real batch sizes before pruning patches.
- **Decoding latency:** The current generator recomputes its whole answer prefix at every step. A decoder key/value cache is a later optimization once OT quality is established. Compare matched-model latency, not just Sinkhorn time.
- **Short question / many patches:** Extreme rectangular transport can route most patch mass to the question dustbin. Monitor local matched mass and gate distributions; consider patch pooling or top-K question-relevant patches only if full-patch OT is too slow or diffuse.
- **Visual grounding limits:** OT aligns co-occurring fragments; it does not itself solve relations, counting, or fine spatial reasoning. The paper's own qualitative analysis notes these failures. A gain on object/attribute questions may coexist with losses on relational questions.
- **No guarantee of gain:** The source paper proves retrieval performance for its tested setup. GQA generation requires independent evidence. Stop or revise OT fusion if the controlled ablation fails to improve the held-out development results.

## References

- Local paper: `docs/2603.14349v1.pdf`; [arXiv HTML version 1](https://arxiv.org/html/2603.14349v1), especially Sections 3, 4.3 and Appendix B1.
- [Authors' OMIT implementation](https://github.com/ppanzx/OMIT) for implementation cross-checks. Its retrieval training task and encoders differ from this VQA repository.
