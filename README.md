# SAN-VQA

Visual question answering with a frozen ViT Base image encoder, BERT Base text
embeddings, a Stacked Attention Network (SAN), a Transformer decoder, and
autoregressive answer generation.

## Setup

```bash
pip install -r requirements.txt
```

The default paths expect GQA CSV files under `data/gqa_dataset` and images under
`data/gqa_dataset/images`. Run `python train.py --help` to see path overrides.

## Train

```bash
python train.py \
  --train_csv_path data/gqa_dataset/train.csv \
  --dev_csv_path data/gqa_dataset/val.csv \
  --img_path data/gqa_dataset/images \
  --model_path data/gqa_model
```

Training writes `last.pt`, the best generated-F1 checkpoint as `best.pt`, a JSONL
metric history, and an evaluation plot. Resume with `--resume path/to/last.pt`.

## Evaluate

```bash
python test.py --checkpoint data/gqa_model/best.pt --split dev
```

## Predict

```bash
python predict.py \
  --checkpoint data/gqa_model/best.pt \
  --image path/to/image.jpg \
  --question "What color is the car?"
```

## Diagnose training

`diagnose_training.py` reports overfitting, output collapse, and sensitivity to
shuffled images or questions.

```bash
python diagnose_training.py \
  --checkpoint data/gqa_model/best.pt \
  --dev_csv_path data/gqa_dataset/val.csv \
  --dev_img_path data/gqa_dataset/images
```

## Tests

```bash
python -m unittest discover -s tests
```
