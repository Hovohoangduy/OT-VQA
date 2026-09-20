# Source logic review

The primary blocker was an invalid training/generation contract. The original `test.py` passed the reference answer to the model in training mode and decoded logits in parallel. The model had no autoregressive inference branch. A low training loss therefore did not establish that it could generate an answer from an image and question.

## Findings corrected

| Area | Original behavior and consequence | Correction |
| --- | --- | --- |
| Answer training | Input IDs were also the same-position targets, including BOS. | Feed `ids[:, :-1]` and predict `ids[:, 1:]`, including EOS. |
| Decoder inputs | Answer embeddings were encoder memory, while repeated image/question vectors were decoder targets. Full answers were visible through unmasked cross-attention. | Use image/question context as memory and answer prefixes as targets; apply causal and padding masks to answer self-attention. |
| Inference | No generation loop; even evaluation required the answer. | Start from BOS, append predicted tokens, stop at EOS, and pad completed rows independently. `predict.py` accepts one image/question. |
| Attention heads | Head output was directly reshaped from `[B,H,T,D]` to `[B,T,H*D]`, mixing token/head positions. | Transpose head and token dimensions before merging. Cross-attention also handles different memory and answer lengths. |
| Images | ViT's `[B,N,D]` hidden states were reshaped as if they were channel-first feature maps. | Preserve token/feature order. |
| Pixels | `ToTensor()` already scaled pixels to `[0,1]`; the image processor rescaled them again. | Disable the second rescaling and retain processor normalization. |
| Frozen vision encoder | Calling `model.train()` re-enabled its dropout although its parameters were frozen. | Keep the frozen ViT encoder in evaluation mode. |
| Stacked attention | All layers referenced one shared object, and each read the original question instead of the preceding context. | Instantiate separate layers and feed each updated context to the next. |
| Question encoding | LSTM summarized fixed-length sequences after their padding tokens. | Pack by attention-mask lengths and summarize valid question tokens. |
| Batch sizes | Model reshape used CLI batch size; training/evaluation skipped incomplete batches. Single-image inference could fail. | Infer dimensions from actual tensors and process every batch. |
| Loss | Mean cross-entropy was divided again by the number of non-padding targets; a progress print could read an undefined/stale loss. | Use mean cross-entropy once with the tokenizer's PAD ID; print the current loss after computing it. |
| Metrics | Manually joined vocabulary entries left tokenizer subword markers in predictions; F1 discarded repeated words. Batch averaging also counted skipped batches. | Decode with the tokenizer, use token counts, and weight metrics by example count and loss by target-token count. |
| Preprocessing | Questions were written into a misspelled `quesion` column, which the dataset did not read. Test preprocessing differed. | Normalize English text consistently in the `question` column for training, evaluation, and prediction. |
| Language-specific code | The dataset, encoder attributes, normalizer, CLI, and optional segmenter dependency included Vietnamese-specific paths. | Support English only with `bert-base-uncased` by default, whitespace normalization, one generic `VQADataset`, and no language flag or Vietnamese segmenter dependency. Reject Vietnamese encoder names before loading weights. |
| Imports | Several modules parsed CLI arguments or downloaded tokenizers at import time; CSV conversion executed on import. | Parse arguments and execute conversion only in entrypoints; datasets do not load tokenizers. |
| CSV/image loading | Every entrypoint loaded all CSV splits; dataset image root depended on global CLI state. GQA lacked `anno_id`. | Load only the required split, pass image roots explicitly, and synthesize optional annotation IDs. |
| Alternative JSON answers | Alternative correct answers were concatenated into one target. | Choose the first annotation as one valid training answer. This is a single-reference policy, not full multi-reference evaluation. |
| Checkpoints | Bare state dictionaries lacked model/preprocessing settings; loading lacked device mapping. | Versioned checkpoints store model dimensions, encoder names, English preprocessing settings, and weights; load strictly with device mapping. Compatible English checkpoints migrate their old module keys to the generic encoder names. Legacy bare state dictionaries receive an explicit retraining error. |
| GQA export | New CSVs used bare image filenames despite storing images in split folders; generated corpus included held-out text. | Export split-relative image paths and annotation IDs, and build corpus from training text only. Existing data is left as previously downloaded. |
| QuMLAG | If only one modality supplied a padding mask, combined memory received a mask with the wrong length. | Infer missing masks and concatenate both modalities' masks. |
| MLPAG | Extended OCR/scene IDs could be passed directly into fixed vocabulary embeddings during teacher forcing. | Map copied scene IDs to vocabulary IDs before embedding while preserving ordinary PAD IDs. |
| M4C causality | Context tokens could read future answer tokens and relay them to earlier positions in later encoder layers. | Block all context-to-answer attention and future answer-to-answer attention. |
| M4C copying | Inference discarded all OCR pointer scores, omitted the final prediction, and could not embed copied IDs. | Select over vocabulary plus OCR positions, feed copied OCR features into subsequent steps, and return all predicted IDs and corresponding step scores. |
| Alternative generation | Finished rows continued producing tokens, and custom decode lengths could exceed positional embeddings. | Pad finished rows, suppress PAD/BOS for unfinished predictions, and validate lengths. BERT question padding is separate from M4C answer padding. |
| Synthetic runners | Targets were unrelated to answer input prefixes; printed output resembled real training/evaluation. | Shift synthetic targets consistently, include EOS, and identify runners as synthetic smoke tests without checkpoint training. |

## Validation and limits

Offline regression tests use small locally initialized BERT and ViT models. They verify gradients, shifted targets, causal isolation, image processing, one-image operation, EOS handling, partial batches, checkpoint round trips and English-key migration, English whitespace normalization and rejection of Vietnamese encoder names, conversion, and alternative model masks/copy generation. One optimization test teaches a tiny model the answer `red` and then generates `red` without supplying its reference answer. This verifies the training/generation contract; it does not measure real VQA quality. In the current workspace, `python -m unittest discover -s tests -q` passes 29 tests with one skipped.

Both alternative CLI runners also completed all five model paths with small CPU settings. All repository Python files were parsed, and imports of training, evaluation, prediction, model, dataset, and conversion modules succeeded with an unrelated CLI argument present. These checks do not establish production accuracy or replace a controlled held-out evaluation; the available GQA training bottleneck measurements and their limits are documented in the implementation plan. Full-size retraining and CUDA execution were not performed.

The main model uses ViT with selectable fusion and has no OCR extraction or copy mechanism. Its ability to answer questions about scene text remains limited. The re-implemented models are architecture prototypes, not complete reproductions of published training pipelines. Their synthetic metrics are not accuracy measurements. In particular, M4C copied output IDs must be resolved to the caller's OCR strings; feature inputs alone do not contain that text mapping.

Long answers and questions remain truncated at configured limits. Checkpoint resume, best-development-checkpoint selection, and multi-reference answer scoring are not implemented. The downloader records a dataset revision as metadata but does not pin requests to that revision; its reproducibility is limited by source changes. Existing GQA `corpus.txt` includes held-out text, so rebuild it from training rows before using it for vocabulary learning or training.

## Documentation consulted

ViT expects its processor's pixel preparation; rescaling can be disabled for already scaled tensors. See [Hugging Face ViT documentation](https://huggingface.co/docs/transformers/model_doc/vit). The supported default text model is English BERT. Implementation conclusions above are based on repository source and regression tests.
