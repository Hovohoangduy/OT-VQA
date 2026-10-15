import argparse

from PIL import Image
import torch

from utils.checkpoint import load_model
from utils.data_processing import preprocess_text
from utils.device import resolve_device
from utils.vqa_dataset import build_question


def main():
    """
    Inference / Generation Flow Pipeline:
        1. Initialize Sequence -> [BOS]
        2. Encode -> Memory (Images & Questions)
        3. Loop until [EOS] or max_len:
             a. Decoder(Memory, Current Sequence) -> Logits
             b. Argmax Logits -> Next Token ID
             c. Append Next Token ID to Sequence
        4. Decode Token IDs -> String (Clean up special tokens & underscores)
    """
    parser = argparse.ArgumentParser(description="Generate an answer from one image and question")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--caption", default=None,
                        help="Figure caption, required by checkpoints trained with --figure_context")
    parser.add_argument("--figure_type", default=None,
                        help="Figure type, required by checkpoints trained with --figure_context")
    parser.add_argument("--answer_options", default=None, help="JSON list of label/text objects")
    parser.add_argument("--qa_pair_type", default=None)
    parser.add_argument("--compound", default=None)
    parser.add_argument("--figs_numb", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"],
                        default="auto")
    args = parser.parse_args()
    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device)
    with Image.open(args.image) as source:
        display_image = source.convert("RGB")
        native = getattr(model, "model_family", "legacy") == "qwen3_vl"
        image = [display_image] if native else model.image_transform(display_image).unsqueeze(0).to(device)
    question = preprocess_text(args.question)
    if model.figure_context:
        if not args.caption or not args.figure_type:
            parser.error("This checkpoint requires --caption and --figure_type")
    query = build_question(question, qa_prompt=False, answer_options=args.answer_options)
    question = build_question(question, qa_pair_type=args.qa_pair_type, caption=args.caption,
                              figure_type=args.figure_type, compound=args.compound, figs_numb=args.figs_numb,
                              figure_context=model.figure_context, prompt_version=model.prompt_version,
                              answer_options=args.answer_options)
    if native:
        question = {"prompt": question, "query": query}
    result = model.generate(image, [question])
    print(model.answers_from_ids(result)[0])


if __name__ == "__main__":
    main()
