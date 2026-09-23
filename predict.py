import argparse
import json

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
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"],
                        default="auto")
    args = parser.parse_args()
    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device)
    with Image.open(args.image) as source:
        display_image = source.convert("RGB")
        image = Config.transforms(display_image).unsqueeze(0).to(device)
    question = preprocess_text(args.question)
    diagnostics = args.diagnostics
    result = model.generate(image, [question], return_diagnostics=diagnostics)
    ids = result.generated_ids if diagnostics else result
    print(model.answers_from_ids(ids)[0])
    if diagnostics:
        payload = {"fusion": model.fusion_type}
        if result.fusion_output is not None and result.fusion_output.diagnostics is not None:
            payload.update({
                f"fusion_{key}": value.detach().float().mean().item()
                for key, value in result.fusion_output.diagnostics.items()
            })
        print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
