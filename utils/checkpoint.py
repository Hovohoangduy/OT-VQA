from __future__ import annotations

import random
from pathlib import Path

import torch

from configs.config import Config
from model.vqa_model import VQAModel


def _adapt_deit_state_dict(state_dict, target_keys):
    """Bridge the DeiT module-path rename across Transformers releases."""
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
    grad_scaler=None,
    epoch=0, global_step=0, best_metric=None, epochs_without_improvement=0,
    format_version=3, alignment_teacher=None, alignment_config=None,
    negative_queue=None, training_stage=None,
):
    if format_version not in {3, 4}:
        raise ValueError("Checkpoint format_version must be 3 or 4")
    if format_version == 3 and any(
        value is not None
        for value in (alignment_teacher, alignment_config, negative_queue, training_stage)
    ):
        raise ValueError("Training-only alignment state requires checkpoint version 4")
    fusion = getattr(model, "fusion_type", model.model_config.get("fusion"))
    if fusion == "cross_attention":
        architecture = "cross_attention_only_v1"
    elif str(fusion).endswith("_v2"):
        architecture = "ot_evidence_routing_v2"
    else:
        architecture = "ot_evidence_routing_v1"
    payload = {
        "format_version": format_version,
        "architecture": architecture,
        "model_state_dict": model.state_dict(),
        "model_config": model.model_config,
        "text_model": text_model,
        "image_model": image_model,
        # Cached-feature training intentionally does not instantiate frozen
        # backbones. Deployment reloads those weights from their pretrained IDs.
        "encoders_omitted": bool(getattr(model, "skip_encoders", False)),
        "encoder_revisions": {
            "text": (
                getattr(model.question_encoder.text_encoder.config, "_commit_hash", None)
                if getattr(model, "question_encoder", None) is not None
                and getattr(model.question_encoder, "text_encoder", None) is not None
                and hasattr(model.question_encoder.text_encoder, "config")
                else None
            ),
            "image": (
                getattr(model.image_model.model.config, "_commit_hash", None)
                if getattr(model, "image_model", None) is not None
                and getattr(model.image_model, "model", None) is not None
                and hasattr(model.image_model.model, "config")
                else None
            ),
        },
        "preprocessing": {
            "max_question_length": Config.MAX_LEN_QUES,
            "max_answer_length": Config.MAX_LEN_ANS,
        },
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "grad_scaler_state_dict": (
            grad_scaler.state_dict() if grad_scaler is not None else None
        ),
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
    if format_version == 4:
        payload.update({
            "alignment_teacher_state_dict": (
                alignment_teacher.state_dict() if alignment_teacher is not None else None
            ),
            "alignment_config": alignment_config,
            "negative_queue_state": (
                negative_queue.state_dict() if negative_queue is not None else None
            ),
            "training_stage": training_stage,
        })
    return payload


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
    if version not in {3, 4}:
        raise ValueError("Only checkpoint versions 3 and 4 are supported")
    fusion = checkpoint.get("model_config", {}).get("fusion")
    allowed = {
        "cross_attention": "cross_attention_only_v1",
        "ot_evidence_routing": "ot_evidence_routing_v1",
        "softmax_evidence_routing": "ot_evidence_routing_v1",
        "ot_evidence_routing_v2": "ot_evidence_routing_v2",
        "softmax_evidence_routing_v2": "ot_evidence_routing_v2",
    }
    if fusion not in allowed:
        raise ValueError(
            f"Checkpoint fusion {fusion!r} was removed or is unsupported"
        )
    expected_architecture = allowed[fusion]
    if checkpoint.get("architecture") != expected_architecture:
        raise ValueError(
            f"Checkpoint architecture does not match fusion {fusion!r}; "
            f"expected {expected_architecture!r}"
        )
    return checkpoint


def load_model(checkpoint_path, device):
    checkpoint = read_checkpoint(checkpoint_path, device)
    model_config = dict(checkpoint.get("model_config", {}))
    model = VQAModel(
        text_model=checkpoint["text_model"], image_model=checkpoint["image_model"],
        **model_config,
    )
    state = _adapt_deit_state_dict(checkpoint["model_state_dict"], model.state_dict())
    if checkpoint.get("encoders_omitted", False):
        incompatible = model.load_state_dict(state, strict=False)
        allowed_missing = (
            "image_model.model.",
            "question_encoder.text_encoder.",
        )
        unexpected = list(incompatible.unexpected_keys)
        disallowed_missing = [
            key for key in incompatible.missing_keys
            if not key.startswith(allowed_missing)
        ]
        if unexpected or disallowed_missing:
            raise RuntimeError(
                "Cached-feature checkpoint has incompatible student weights: "
                f"missing={disallowed_missing}, unexpected={unexpected}"
            )
    else:
        model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()
    return model


def restore_training_state(checkpoint, model, optimizer, scheduler, grad_scaler=None):
    if checkpoint.get("format_version") not in {3, 4}:
        raise ValueError("Only version-3/4 checkpoints contain resumable training state")
    state = _adapt_deit_state_dict(checkpoint["model_state_dict"], model.state_dict())
    model.load_state_dict(state, strict=True)
    if checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if grad_scaler is not None and checkpoint.get("grad_scaler_state_dict") is not None:
        grad_scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
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


def restore_alignment_state(checkpoint, teacher, negative_queue=None):
    """Restore the training-only teacher and queue from a version-4 checkpoint."""
    if checkpoint.get("format_version") != 4:
        raise ValueError("OT alignment training can resume only from version 4")
    state = checkpoint.get("alignment_teacher_state_dict")
    if state is None:
        raise ValueError("Version-4 checkpoint does not contain an alignment teacher")
    teacher.load_state_dict(state, strict=True)
    if negative_queue is not None:
        negative_queue.load_state_dict(checkpoint.get("negative_queue_state"))
    return checkpoint.get("training_stage")


def load_student_initialization(checkpoint_path, model):
    """Strictly copy only student weights for paired common-initialization runs."""
    checkpoint = read_checkpoint(checkpoint_path, torch.device("cpu"))
    source_config = checkpoint.get("model_config", {})
    target_config = model.model_config
    comparable_source = dict(source_config)
    comparable_target = dict(target_config)
    routing_fusions = {
        "ot_evidence_routing", "softmax_evidence_routing",
        "ot_evidence_routing_v2", "softmax_evidence_routing_v2",
    }
    if (comparable_source.get("fusion") in routing_fusions or
            comparable_target.get("fusion") in routing_fusions):
        from model.ot_routing import OTEvidenceRoutingConfig
    if comparable_source.get("fusion") in routing_fusions:
        comparable_source["routing_config"] = OTEvidenceRoutingConfig.from_dict(
            comparable_source.get("routing_config")
        ).to_dict()
    if comparable_target.get("fusion") in routing_fusions:
        comparable_target["routing_config"] = OTEvidenceRoutingConfig.from_dict(
            comparable_target.get("routing_config")
        ).to_dict()
    if (
        comparable_source.get("fusion") in routing_fusions
        and comparable_target.get("fusion") in routing_fusions
        and comparable_source.get("fusion", "").endswith("_v2")
        == comparable_target.get("fusion", "").endswith("_v2")
    ):
        comparable_source["fusion"] = "evidence_routing_control"
        comparable_target["fusion"] = "evidence_routing_control"
    if comparable_source != comparable_target:
        raise ValueError("Student initialization checkpoint model_config does not match")
    state = _adapt_deit_state_dict(checkpoint["model_state_dict"], model.state_dict())
    model.load_state_dict(state, strict=True)
