# OT-VQA

Visual question answering with a frozen ViT image encoder, BERT question tokens,
partial optimal transport fusion, and autoregressive answer generation. The
original Stacked Attention Network (SAN) remains available as a baseline. See
[the implementation and evaluation plan](docs/optimal_transport_vqa_plan.md).

## Setup

```bash
pip install -r requirements.txt
```

The default paths expect GQA CSV files under `data/gqa_dataset` and images under
`data/gqa_dataset/images`. Run `python train.py --help` to see path overrides.

To build a larger local GQA subset with the original one-question-per-image format:

```bash
python -m utils.download_gqa --train-images 10000 --val-images 500 \
  --test-images 500
```

Keep the held-out test split for the final comparison. Dataset downloads require
network access and can be large; choose counts that fit your compute budget.
The downloader spaces dataset viewer requests, honors HTTP 429 retry delays, and
saves metadata pages in `<output>/.row_cache`. Rerun with the same `--output` to
reuse completed pages. If rate limiting persists, set `HF_TOKEN` in your shell
and try `--metadata-workers 1 --request-interval 2`. `--workers` controls image
downloads separately.

## Train

For NVIDIA GPU training on Windows, install a CUDA-enabled `torch` and matching
`torchvision` in your active environment using the
[official PyTorch installer](https://pytorch.org/get-started/locally/). Installing
`requirements.txt` alone does not guarantee a CUDA build. Check your installation
in PowerShell:

```powershell
conda activate ot
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

`CUDA available` must print `True`. Then train with CUDA explicitly:

```powershell
python train.py --fusion ot --device cuda --train_csv_path data/gqa_dataset/train.csv --dev_csv_path data/gqa_dataset/val.csv --img_path data/gqa_dataset/images --model_path data/gqa_model
```

Training prints `Training on device: cuda`. If CUDA is unavailable, `--device cuda`
stops with an error instead of training on CPU. Reduce `--batch_size` from its
default of 4 if the GPU runs out of memory. `--device auto` also uses CUDA when
available, but can fall back to another device.

Training writes `last.pt`, the best generated-F1 checkpoint as `best.pt`, a JSONL
metric history, an evaluation plot, and `run_config.json` with dataset hashes.
Resume with `--resume path/to/last.pt`.
New runs default to OT. For a matched SAN baseline, run the same command with
`--fusion san` and a different `--model_path`. Set `--ot_dustbin_mass 0` for a
full-OT ablation. Partial OT defaults to mass `0.2`, Sinkhorn epsilon `0.05`,
20 iterations, and dustbin-to-dustbin cost `1.0`. These values require
validation on your dataset.

## Evaluate

```bash
python test.py --checkpoint data/gqa_model/best.pt --split dev \
  --predictions_csv data/gqa_model/dev_predictions.csv \
  --report_json data/gqa_model/dev_report.json
```

Evaluation reports generated exact match, token F1, examples per second, and
peak CUDA memory. If your CSV has a `question_type` column, the prediction export
includes it. After evaluating SAN and OT on the *same* held-out examples, compare
their prediction files with paired bootstrap intervals:

```bash
python -m utils.compare_predictions \
  --san data/san_model/dev_predictions.csv \
  --ot data/gqa_model/dev_predictions.csv \
  --output-json data/ot_comparison.json
```

Pass multiple files after `--san` and `--ot` in the same seed order when you
have repeated runs. The paper's results are for image-text retrieval; VQA gains
must be established with this repository's generated-answer evaluation.

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
