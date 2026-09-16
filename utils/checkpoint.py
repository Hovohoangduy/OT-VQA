import torch

from model.vqa_model import VQAModel


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 2:
        raise ValueError("Legacy checkpoint was trained with the incorrect decoder/objective. "
                         "Retrain with the corrected train.py before generating answers.")
    model = VQAModel(text_model=checkpoint["text_model"], image_model=checkpoint["image_model"],
                     **checkpoint.get("model_config", {})).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint.get("language", "vi")
