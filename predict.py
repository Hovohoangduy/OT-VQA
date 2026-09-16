import argparse
import json

from PIL import Image
import torch

from configs.config import Config
from utils.checkpoint import load_model
from utils.data_processing import preprocess_text
from utils.device import resolve_device
from utils.transport_visualization import save_transport_diagnostics


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
    parser.add_argument("--diagnostics_output", default=None,
                        help="Optional path for an OT plan/cost/marginal figure")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"],
                        default="auto")
    args = parser.parse_args()
    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device)
    with Image.open(args.image) as source:
        display_image = source.convert("RGB")
        image = Config.transforms(display_image).unsqueeze(0).to(device)
    question = preprocess_text(args.question)
    diagnostics = args.diagnostics or args.diagnostics_output is not None
    result = model.generate(image, [question], return_diagnostics=diagnostics)
    ids = result.generated_ids if diagnostics else result
    print(model.answers_from_ids(ids)[0])
    if diagnostics:
        transport = result.transport
        if transport is None:
            if args.diagnostics_output:
                raise ValueError("diagnostics_output requires a Balanced OT or UOT checkpoint")
            print(json.dumps({"fusion": "san"}))
        else:
            payload = {
                "fusion": model.fusion_type,
                "transport_cost": transport.transport_cost.item(),
                "entropy": transport.entropy.item(),
                "matched_mass": transport.matched_mass.item(),
                "unmatched_mass": transport.unmatched_mass.item(),
                "residual": transport.residual.item(),
                "iterations": transport.iterations.item(),
                "converged": bool(transport.converged.item()),
            }
            print(json.dumps(payload, sort_keys=True))
            if args.diagnostics_output:
                encoded = model.question_encoder.tokenizer(
                    [question], max_length=Config.MAX_LEN_QUES,
                    truncation=True, padding=True, return_tensors="pt",
                )
                tokens = model.question_encoder.tokenizer.convert_ids_to_tokens(
                    encoded["input_ids"][0].tolist()
                )
                save_transport_diagnostics(
                    transport, args.diagnostics_output,
                    image=display_image, question_tokens=tokens,
                )


if __name__ == "__main__":
    main()
