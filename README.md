# OT-VQA

Train a Vietnamese visual question answering model, evaluate answers generated without references, and predict from one image/question.

```bash
python -m pip install -r requirements.txt
python train.py --train_csv_path data/csv/ViTextVQA_train.csv --img_path data/images/st_images --model_path data
python test.py --dev_csv_path data/csv/ViTextVQA_dev.csv --img_path data/images/st_images --model_path data
python predict.py --checkpoint data/vi_text.pt --image path/to/image.jpg --question "Trong ảnh có gì?"
```

The first run downloads DeiT and PhoBERT. Vietnamese questions and answers are word segmented consistently during training and prediction. CSVs require `image`, `question`, and `answer`; `anno_id` is optional. Image paths are relative to `--img_path`. Training needs only its training CSV; evaluation loads only its selected split. Use `--split test --test_csv_path ...` to score a labelled test set. Prediction needs no answer or CSV.

**Retrain old checkpoints.** The previous code used an incorrect objective and exposed full answers through cross-attention. Corrected training predicts the next token from a shifted answer prefix with causal attention. New checkpoints include model settings and preprocessing language. Loading an old bare state dictionary raises an explanatory error.

Training EM/F1 are teacher-forced diagnostics. `test.py` calculates EM/F1 from autoregressive generation and reports teacher-forced loss separately. No quality claim can be made without retraining and evaluating on real held-out data.

## Existing English GQA subset

The included subset uses bare image filenames and separate image folders. Use an English text encoder and English preprocessing:

```bash
python train.py --train_csv_path data/gqa_dataset/train.csv --img_path data/gqa_dataset/images/train --text_model bert-base-uncased --language en --model_path data/gqa_model
python test.py --dev_csv_path data/gqa_dataset/val.csv --img_path data/gqa_dataset/images/val --model_path data/gqa_model
python predict.py --checkpoint data/gqa_model/vi_text.pt --image data/gqa_dataset/images/test/IMAGE_ID.jpg --question "What color is it?"
```

PhoBERT is a Vietnamese encoder; do not use the Vietnamese defaults for this English dataset. The downloader now writes split-relative image paths such as `train/123.jpg`, so newly downloaded CSVs use `--img_path data/gqa_dataset/images` for all splits. Its generated corpus contains training text only; the existing corpus predates this correction.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests create tiny local BERT/DeiT models and do not download pretrained weights. They check shifted targets, causal isolation, gradients, pixel processing, EOS/padding behavior, partial batches, checkpoint loading, data conversion, and alternative model masks/copy generation.

Files in `model/re-implement_model` are architecture prototypes with synthetic runners. Their printed losses and metrics use random inputs/targets and do not demonstrate training or model accuracy. They need real tokenizers, visual/OCR feature extraction, datasets, optimizer loops, and checkpoint workflows before use as trained models. M4C generated IDs greater than or equal to `vocab_size` identify OCR positions and must be resolved to OCR strings by the caller. The main DeiT/SAN model has no OCR extraction or copy head, which limits scene-text answering.

See [the source review](docs/source_review.md) for the problems found and verification details.

The current main model uses SAN fusion; it does not yet implement an Optimal Transport
solver. See the [Question-Conditioned UOT implementation plan](docs/optimal_transport_implementation_plan.md)
for the ordered design, interfaces, tests, and experiment matrix for adding real OT fusion.
