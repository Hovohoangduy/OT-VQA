import argparse

from PIL import Image
import torch

from configs.config import Config
from utils.checkpoint import load_model
from utils.data_processing import preprocess_text


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
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, language = load_model(args.checkpoint, device)
    with Image.open(args.image) as source:
        image = Config.transforms(source.convert("RGB")).unsqueeze(0).to(device)
    question = preprocess_text(args.question, language)
    ids = model.generate(image, [question])
    print(model.answers_from_ids(ids)[0].replace("_", " "))


if __name__ == "__main__":
    main()
