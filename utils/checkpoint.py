from __future__ import annotations

import random
from pathlib import Path

import torch

from configs.config import Config
from model.vqa_model import VQAModel


MODEL_CONFIG_KEYS = {
    "vocab_size", "output_size", "d_model", "num_heads", "ffn_hidden",
    "drop_prob", "num_layers", "num_att_layers", "mode",
    "freeze_answer_embeddings", "fusion", "ot_epsilon", "ot_iterations",
    "ot_dustbin_mass", "ot_dustbin_cost", "max_answer_tokens",
}


def _adapt_vision_state_dict(state_dict, target_keys):
    """Bridge the ViT module-path rename across Transformers releases."""
    target_keys = set(target_keys)
    old_marker = "image_model.model.encoder.layer."
    new_marker = "image_model.model.layers."
    source_uses_new = any(key.startswith(new_marker) for key in state_dict)
    target_uses_new = any(key.startswith(new_marker) for key in target_keys)
    if source_uses_new == target_uses_new:
        return state_dict
    old_to_new = {
        ".attention.attention.query.": ".attention.q_proj.",
        ".attention.attention.key.": ".attention.k_proj.",
        ".attention.attention.value.": ".attention.v_proj.",
        ".attention.output.dense.": ".attention.o_proj.",
        ".intermediate.dense.": ".mlp.fc1.",
        ".output.dense.": ".mlp.fc2.",
    }
    converted = {}
    for key, value in state_dict.items():
        updated = key
        if source_uses_new and key.startswith(new_marker):
            updated = key.replace(new_marker, old_marker, 1)
            for old, new in old_to_new.items():
                updated = updated.replace(new, old)
        elif not source_uses_new and key.startswith(old_marker):
            updated = key.replace(old_marker, new_marker, 1)
            for old, new in old_to_new.items():
                updated = updated.replace(old, new)
        converted[updated] = value
    return converted


def checkpoint_payload(
    model, text_model, image_model, optimizer=None, scheduler=None,
    epoch=0, global_step=0, best_metric=None, epochs_without_improvement=0,
):
    return {
        "format_version": 4,
        "model_state_dict": model.state_dict(),
        "model_config": model.model_config,
        "text_model": text_model,
        "image_model": image_model,
        "encoder_revisions": {
            "text": getattr(model.question_encoder.text_encoder.config, "_commit_hash", None),
            "image": getattr(model.image_model.model.config, "_commit_hash", None),
        },
        "preprocessing": {
            "max_question_length": Config.MAX_LEN_QUES,
            "max_answer_length": getattr(model, "max_answer_tokens", Config.MAX_LEN_ANS),
        },
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": best_metric,
        "epochs_without_improvement": int(epochs_without_improvement),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "mps_rng_state": (torch.mps.get_rng_state()
                          if torch.backends.mps.is_available() else None),
        "python_rng_state": random.getstate(),
    }


def save_checkpoint(path, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint_payload(**kwargs), temporary)
    temporary.replace(path)


def read_checkpoint(checkpoint_path, device):
    # Stage on CPU so loading a large checkpoint does not temporarily duplicate
    # the complete state dictionary in limited CUDA/MPS device memory.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary")
    version = checkpoint.get("format_version")
    if version not in {2, 3, 4}:
        raise ValueError(
            "Legacy checkpoint was trained with the incorrect decoder/objective. "
            "Retrain with the corrected train.py before generating answers."
        )
    return checkpoint


def load_model(checkpoint_path, device):
    checkpoint = read_checkpoint(checkpoint_path, device)
    stored_config = dict(checkpoint.get("model_config", {}))
    fusion = stored_config.get("fusion", "san")
    if fusion not in {"san", "ot"}:
        raise ValueError(f"Unknown checkpoint fusion architecture: {fusion}")
    model_config = {
        key: value for key, value in stored_config.items() if key in MODEL_CONFIG_KEYS
    }
    model = VQAModel(
        text_model=checkpoint["text_model"], image_model=checkpoint["image_model"],
        **model_config,
    )
    state = _adapt_vision_state_dict(checkpoint["model_state_dict"], model.state_dict())
    state = _adapt_legacy_text_keys(state)
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()
    return model


def _adapt_legacy_text_keys(state_dict):
    """Map legacy text-branch modules to the current English names."""
    converted = {}
    for key, value in state_dict.items():
        updated = key
        parts = key.split(".")
        if len(parts) > 2 and parts[0] == "ques_model":
            if parts[1] == "lstm":
                updated = ".".join(["question_encoder", *parts[1:]])
            else:
                updated = ".".join(["question_encoder", "text_encoder", *parts[2:]])
        elif len(parts) > 2 and parts[0] == "ans_model":
            updated = ".".join(["answer_embedding", "token_embeddings", *parts[2:]])
        converted[updated] = value
    # Leave unknown legacy entries intact so strict loading still reports any
    # genuine architecture mismatch rather than silently discarding parameters.
    return converted


def restore_training_state(checkpoint, model, optimizer, scheduler):
    if checkpoint.get("format_version") not in {3, 4}:
        raise ValueError("Only version-3/4 checkpoints contain resumable training state")
    state = _adapt_vision_state_dict(checkpoint["model_state_dict"], model.state_dict())
    state = _adapt_legacy_text_keys(state)
    model.load_state_dict(state, strict=True)
    if checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if checkpoint.get("torch_rng_state") is not None:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in checkpoint["cuda_rng_state"]]
        )
    if torch.backends.mps.is_available() and checkpoint.get("mps_rng_state") is not None:
        torch.mps.set_rng_state(checkpoint["mps_rng_state"].cpu())
    if checkpoint.get("python_rng_state") is not None:
        random.setstate(checkpoint["python_rng_state"])
    return (int(checkpoint.get("epoch", 0)), int(checkpoint.get("global_step", 0)),
            checkpoint.get("best_metric"))
