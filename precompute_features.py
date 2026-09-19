"""Precompute frozen ViT/DeiT and contextual question-token features."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from configs.config import Config
from model.features_extraction import ImageEmbedding, QuestionEmbedding
from utils.data_processing import load_dataframe
from utils.device import resolve_device
from utils.feature_cache import file_fingerprint, write_feature_cache
from utils.vqa_dataset import VQADataset


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--img_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text_model", default=Config.text_model,
                        help="English Hugging Face tokenizer and text encoder")
    parser.add_argument("--image_model", default=Config.image_model)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"],
                        default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")

    device = resolve_device(args.device)
    print(f"Precomputing on device: {device}")
    frame = load_dataframe(args.csv)
    dataset = VQADataset(frame, Config.transforms, args.img_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    image_encoder = ImageEmbedding(args.image_model).to(device).eval()
    question_encoder = QuestionEmbedding(model_name=args.text_model).to(device).eval()
    samples = []
    with torch.no_grad():
        for anno_ids, images, questions, answers in loader:
            image_features, _ = image_encoder(images.to(device), anno_ids)
            question_features, question_mask, question_ids = question_encoder.encode_tokens(questions)
            valid_lengths = question_ids.ne(question_encoder.tokenizer.pad_token_id).sum(1)
            for row in range(len(questions)):
                length = int(valid_lengths[row])
                samples.append({
                    "anno_id": anno_ids[row].item() if torch.is_tensor(anno_ids) else anno_ids[row],
                    "image_features": image_features[row].detach().cpu().half(),
                    "question_features": question_features[row, :length].detach().cpu().half(),
                    "question_padding_mask": question_mask[row, :length].detach().cpu(),
                    "question": questions[row],
                    "answer": answers[row],
                })
    manifest = {
        "dataset_path": str(Path(args.csv).resolve()),
        "dataset_fingerprint": file_fingerprint(args.csv),
        "text_model": args.text_model,
        "image_model": args.image_model,
        "text_revision": getattr(question_encoder.text_encoder.config, "_commit_hash", None),
        "image_revision": getattr(image_encoder.model.config, "_commit_hash", None),
        "question_max_length": Config.MAX_LEN_QUES,
        "tokenizer": {
            "class": type(question_encoder.tokenizer).__name__,
            "vocab_size": len(question_encoder.tokenizer),
            "padding_side": question_encoder.tokenizer.padding_side,
            "truncation_side": question_encoder.tokenizer.truncation_side,
        },
        "image_processor": {
            "class": type(image_encoder.process).__name__,
            "size": _json_value(image_encoder.process.size),
            "crop_size": _json_value(getattr(image_encoder.process, "crop_size", None)),
            "image_mean": _json_value(image_encoder.process.image_mean),
            "image_std": _json_value(image_encoder.process.image_std),
        },
        "special_token_policy": (
            "question boundary tokens masked; visual CLS/distillation prefix tokens "
            "removed in fusion"
        ),
        "visual_source_shape": list(samples[0]["image_features"].shape) if samples else None,
        "question_hidden_size": samples[0]["question_features"].shape[-1] if samples else None,
        "annotation_ids": [str(sample["anno_id"]) for sample in samples],
    }
    write_feature_cache(args.output, samples, manifest)
    print(f"Cached {len(samples)} samples in {args.output}")


if __name__ == "__main__":
    main()
