"""Checkpoint metadata must work with PyTorch's restricted weights-only loader."""
from types import SimpleNamespace
import pickle

import pytest
import torch
from PIL.Image import Resampling
from transformers.utils.quantization_config import QuantizationMethod

from utils.checkpoint import checkpoint_payload, read_checkpoint, save_checkpoint


def checkpoint_model():
    model = torch.nn.Linear(2, 2)
    model.model_family = "qwen3_vl"
    model.model_config = {
        "fusion": "ot", "vlm_revision": "a" * 40, "query_revision": "b" * 40,
        "quantization_config": {"quant_method": QuantizationMethod.BITS_AND_BYTES},
        "processor_config": {
            "nested": [(QuantizationMethod.BITS_AND_BYTES,)],
            "image_processor": {"resample": Resampling.BICUBIC},
            "video_processor": {"resample": Resampling.BICUBIC},
        },
    }
    model.max_question_tokens = 256
    model.max_answer_tokens = 64
    model.image_size = None
    model.image_preprocessing = "native"
    model.figure_context = True
    return model


def test_new_checkpoint_metadata_loads_without_allowlist(tmp_path):
    model = checkpoint_model()
    path = tmp_path / "new.pt"
    save_checkpoint(path, model=model, text_model="q", image_model="v")
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    config = checkpoint["model_config"]
    assert type(config["quantization_config"]["quant_method"]) is str
    assert config["quantization_config"]["quant_method"] == "bitsandbytes"
    assert type(config["processor_config"]["nested"][0][0]) is str
    assert type(config["processor_config"]["image_processor"]["resample"]) is int
    assert config["processor_config"]["video_processor"]["resample"] == int(Resampling.BICUBIC)
    # Saving metadata must not mutate the live model configuration.
    assert model.model_config["quantization_config"]["quant_method"] is QuantizationMethod.BITS_AND_BYTES
    torch.testing.assert_close(checkpoint["model_state_dict"]["weight"], model.weight)


def test_read_old_checkpoint_with_quantization_and_processor_enums(tmp_path):
    model = checkpoint_model()
    payload = checkpoint_payload(model, "q", "v")
    # Simulate the old writer, which saved Transformers enum instances verbatim.
    payload["model_config"] = model.model_config
    path = tmp_path / "old.pt"
    torch.save(payload, path)
    with pytest.raises(pickle.UnpicklingError, match="QuantizationMethod"):
        torch.load(path, weights_only=True, map_location="cpu")
    allowed_before = list(torch.serialization.get_safe_globals())
    checkpoint = read_checkpoint(path, "cpu")
    assert checkpoint["model_config"]["quantization_config"]["quant_method"] == "bitsandbytes"
    assert torch.serialization.get_safe_globals() == allowed_before
    torch.testing.assert_close(checkpoint["model_state_dict"]["weight"], model.weight)


def test_read_checkpoint_still_rejects_unrelated_globals(tmp_path):
    payload = checkpoint_payload(checkpoint_model(), "q", "v")
    payload["model_config"] = {"fusion": "ot"}
    payload["unexpected"] = SimpleNamespace(value=1)
    path = tmp_path / "unsupported.pt"
    torch.save(payload, path)
    with pytest.raises(pickle.UnpicklingError, match="SimpleNamespace"):
        read_checkpoint(path, "cpu")
