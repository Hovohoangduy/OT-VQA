import argparse

from PIL import Image
import torch

from configs.config import Config
from utils.checkpoint import load_model
from utils.data_processing import preprocess_text
from utils.device import resolve_device


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
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"],
                        default="auto")
    args = parser.parse_args()
    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device)
    with Image.open(args.image) as source:
        display_image = source.convert("RGB")
        image = Config.transforms(display_image).unsqueeze(0).to(device)
    question = preprocess_text(args.question)
    result = model.generate(image, [question])
    print(model.answers_from_ids(result)[0])


if __name__ == "__main__":
    main()
