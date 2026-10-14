"""Regression checks for scientific figures, numeric answers and OT memory."""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image, ImageDraw
from torch import nn

from configs.arg_parser import get_args
from diagnose_training import _different_image_sources
from model.features_extraction import ImageEmbedding, QuestionEmbedding
from model.optimal_transport import PartialTransportFusion
from utils.answer_text import restore_numeric_spacing
from utils.image_processing import build_image_transform
from utils.metrics import scivqa_score_pairs
from utils.vqa_dataset import VQADataset, append_figure_context


@pytest.mark.parametrize("decoded,expected", [
    ("0. 5", "0.5"), ("approximately 0. 66", "approximately 0.66"),
    ("- 2. 75", "-2.75"), ("1, 234", "1,234"), ("1. 2 e - 3", "1.2e-3"),
    ("a, b", "a, b"), ("1, 2", "1, 2"), ("1. The value is 2", "1. The value is 2"),
])
def test_numeric_decoding_preserves_content(decoded, expected):
    assert restore_numeric_spacing(decoded) == expected


def test_pad_preserves_figure_borders_and_aspect_ratio():
    figure = Image.new("RGB", (400, 100), "black")
    draw = ImageDraw.Draw(figure)
    draw.rectangle((0, 0, 9, 99), fill="red")
    draw.rectangle((390, 0, 399, 99), fill="blue")
    image = build_image_transform(512, "pad")(figure)
    assert image.shape == (3, 512, 512)
    assert torch.all(image[:, 0, :] == 1)  # white padding above the full figure
    assert image[0, 256, 0] == 1 and image[2, 256, -1] == 1
    assert torch.all(image[:, 256, 256] == 0)


def test_default_transform_matches_legacy_crop():
    from configs.config import Config
    figure = Image.new("RGB", (400, 100), "red")
    assert torch.equal(build_image_transform()(figure), Config.transforms(figure))


def test_prepared_high_resolution_batch_bypasses_processor_crop():
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(model_type="mobilevitv2")

        def forward(self, pixel_values):
            assert pixel_values.shape == (1, 3, 512, 512)
            assert torch.all(pixel_values[:, 0] == 0.3)  # RGB -> BGR
            return SimpleNamespace(last_hidden_state=pixel_values[:, :, ::32, ::32])

    encoder = ImageEmbedding.__new__(ImageEmbedding)
    nn.Module.__init__(encoder)
    encoder.image_size = 512
    encoder.model = Backbone()
    encoder.process = SimpleNamespace(do_flip_channel_order=True, do_normalize=False)
    pixels = torch.tensor([0.1, 0.2, 0.3]).view(1, 3, 1, 1).expand(1, 3, 512, 512)
    features, _ = encoder(pixels)
    assert features.shape == (1, 256, 3)


@pytest.mark.parametrize("dustbin_mass", [0.0, 0.2])
def test_visual_memory_has_finite_gradients_and_ignores_question_padding(dustbin_mass):
    torch.manual_seed(4)
    fusion = PartialTransportFusion(12, 16, iterations=60,
                                    dustbin_mass=dustbin_mass, visual_memory=True)
    images = torch.randn(2, 9, 16, requires_grad=True)
    questions = torch.randn(2, 5, 12, requires_grad=True)
    valid = torch.tensor([[True, True, False, False, False], [True, True, True, True, False]])
    memory, blocked, diagnostics = fusion(images, questions, valid, return_transport=True)
    assert memory.shape == (2, 15, 16)
    assert not blocked[:, -9:].any()
    assert torch.equal(blocked[:, 1:6], ~valid)
    changed = questions.detach().clone()
    changed[~valid] = 1000
    other, _ = fusion(images, changed, valid)
    assert torch.allclose(memory, other, atol=1e-6)
    assert diagnostics["column_residual"].max() < 1e-5
    memory.square().mean().backward()
    for parameter in fusion.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert torch.isfinite(images.grad).all()
    assert torch.all(questions.grad[~valid] == 0)


def test_legacy_fusion_state_dict_still_loads_strictly():
    legacy = PartialTransportFusion(12, 16)
    restored = PartialTransportFusion(12, 16, visual_memory=False)
    restored.load_state_dict(legacy.state_dict(), strict=True)
    assert not any("image_position" in key for key in restored.state_dict())


def test_sci_defaults_and_explicit_baseline_overrides():
    args = get_args(["--dataset", "scivqa"])
    assert args.lr == 1e-5 and args.encoder_lr == 2e-6 and args.warmup_ratio == 0.05
    assert args.freeze_question_encoder is True
    assert args.loss_normalization == "example"
    assert args.image_size == 512 and args.image_preprocessing == "pad"
    assert args.ot_visual_memory is None
    args = get_args(["--dataset", "scivqa", "--lr", "1e-5", "--warmup_ratio", "0",
                     "--selection_metric", "loss", "--image_size", "512", "--image_preprocessing", "pad"])
    assert args.lr == 1e-5 and args.warmup_ratio == 0 and args.selection_metric == "loss"
    assert args.image_size == 512 and args.image_preprocessing == "pad"
    assert get_args(["--dataset", "scivqa", "--loss_normalization", "token"]).loss_normalization == "token"
    assert get_args(["--dataset", "scivqa", "--figure_context"]).figure_context is True
    assert get_args(["--dataset", "scivqa", "--no-freeze_question_encoder"]).freeze_question_encoder is False
    assert get_args(["--dataset", "plantexpert"]).freeze_question_encoder is False


def test_frozen_question_encoder_stays_in_eval_mode_and_has_no_gradients():
    encoder = QuestionEmbedding.__new__(QuestionEmbedding)
    nn.Module.__init__(encoder)
    encoder.text_encoder = nn.Sequential(nn.Linear(3, 3), nn.Dropout(0.5))
    encoder.encoder_frozen = False
    encoder.freeze()
    encoder.train()
    assert encoder.training
    assert not encoder.text_encoder.training
    assert not any(parameter.requires_grad for parameter in encoder.text_encoder.parameters())


def test_frozen_question_encoder_has_no_optimizer_group():
    from train import _build_optimizer

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.question_encoder = nn.Linear(2, 2)
            self.answer_embedding = nn.Linear(2, 2)
            self.decoder = nn.Linear(2, 2)
            self.question_encoder.requires_grad_(False)
            self.answer_embedding.requires_grad_(False)

    model = Model()
    args = SimpleNamespace(lr=1e-5, encoder_lr=2e-6, weight_decay=0.05)
    optimizer = _build_optimizer(model, args)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["group_name"] == "decoder"
    assert optimizer.param_groups[0]["params"] == list(model.decoder.parameters())
    resumed = _build_optimizer(model, args, {"optimizer_state_dict": optimizer.state_dict()})
    resumed.load_state_dict(optimizer.state_dict())


def test_figure_context_uses_csv_metadata_once_and_keeps_question_first(tmp_path):
    import pandas as pd

    Image.new("RGB", (8, 8), "white").save(tmp_path / "figure.png")
    row = {"image": "figure.png", "question": "What is the red value?",
           "answer": "5", "caption": "Accuracy by epoch", "figure_type": "line chart"}
    dataset = VQADataset(pd.DataFrame([row]), img_path=tmp_path, figure_context=True)
    question = dataset[0][2]
    assert question == "What is the red value?\nFigure type: line chart\nFigure caption: Accuracy by epoch"
    assert append_figure_context(question, row["caption"], row["figure_type"]) == question
    assert append_figure_context("What? Figure caption: Existing", row["caption"], "") == (
        "What? Figure caption: Existing")
    with pytest.raises(ValueError, match="caption and figure_type"):
        VQADataset(pd.DataFrame([{"image": "figure.png", "question": "What?"}]),
                   img_path=tmp_path, figure_context=True)


def test_scivqa_rouge_uses_official_stemming():
    class SemanticScorer:
        def score(self, hypotheses, references):
            return (torch.ones(len(hypotheses)),) * 3

    row = scivqa_score_pairs(["dogs"], ["dog"], SemanticScorer())[0]
    assert row["rouge_1_f1"] == 1.0
    assert row["rouge_l_f1"] == 1.0


def test_paper_comparison_aligns_annotation_ids(tmp_path):
    import pandas as pd
    from compare_scivqa import compare

    gold = tmp_path / "gold.csv"
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame({"anno_id": ["a", "b"], "answer": ["dogs", "cats"]}).to_csv(gold, index=False)
    pd.DataFrame({"anno_id": ["b", "a"], "prediction": ["cat", "dog"]}).to_csv(
        predictions, index=False)
    result = compare(gold, predictions, "test")
    assert result["rouge_l_f1"] == 1.0
    assert result["rouge_l_gap_to_paper"] == pytest.approx(0.26)
    pd.DataFrame({"anno_id": ["a", "c"], "prediction": ["dog", "cat"]}).to_csv(
        predictions, index=False)
    with pytest.raises(ValueError, match="Prediction IDs do not match"):
        compare(gold, predictions, "test")


def test_image_counterfactual_always_uses_another_figure():
    names = ["a"] * 7 + ["b"] * 7 + ["c"] * 7
    sources = _different_image_sources(names)
    assert all(names[i] != names[source] for i, source in enumerate(sources))
    with pytest.raises(ValueError, match="distinct figures"):
        _different_image_sources(["a", "a"])


def test_evaluation_reports_question_groups_separately():
    import pandas as pd
    from torch.utils.data import DataLoader, Dataset
    from test import evaluation

    class Examples(Dataset):
        dataset_name = "scivqa"
        data = pd.DataFrame({"qa_pair_type": ["binary", "numerical"]})

        def __len__(self):
            return 2

        def __getitem__(self, i):
            return i, torch.zeros(3, 32, 32), "question", ["yes", "0.5"][i]

    class Model(nn.Module):
        pad_token_id = 0

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))

        def evaluate_batch(self, images, questions, answers, annotations):
            return torch.zeros(2, 1, 3), torch.ones(2, 1, dtype=torch.long), annotations

        def answers_from_ids(self, ids):
            return ["yes", "1.5"]

    class SemanticScorer:
        def score(self, hypotheses, references):
            return (torch.ones(2),) * 3

    result = evaluation(Model(), DataLoader(Examples(), batch_size=2),
                        nn.CrossEntropyLoss(), bert_scorer=SemanticScorer())
    grouped = result["metrics_by_qa_pair_type"]
    assert grouped["binary"]["examples"] == grouped["numerical"]["examples"] == 1
    assert grouped["binary"]["metrics"]["rouge_l_f1"] == 1
    assert grouped["numerical"]["metrics"]["rouge_l_f1"] == 0.5
    assert result["metrics"]["rouge_l_f1"] == 0.75


@pytest.mark.parametrize("smoothing", [0.0, 0.1])
def test_answer_loss_normalization_and_padding(smoothing):
    from utils.losses import AnswerCrossEntropyLoss

    # One easy short answer and one difficult long answer. Their influence
    # should be equal in example mode, despite the threefold length difference.
    logits = torch.tensor([[[0., 7., -8.], [2., -7., 8.]],
                           [[2., 2., 2.], [0., 0., 0.]]], requires_grad=True)
    targets = torch.tensor([[1, -100, -100], [1, 1, 1]])
    losses = torch.nn.functional.cross_entropy(logits, targets, ignore_index=-100,
                                               label_smoothing=smoothing, reduction="none")
    token = AnswerCrossEntropyLoss(-100, smoothing, "token")(logits, targets)
    example = AnswerCrossEntropyLoss(-100, smoothing, "example")(logits, targets)
    assert torch.allclose(token, (losses[0, 0] + losses[1].sum()) / 4)
    assert torch.allclose(example, (losses[0, 0] + losses[1].mean()) / 2)
    more_logits = torch.cat((logits, torch.randn(2, 2, 2)), dim=2)
    more_targets = torch.cat((targets, torch.full((2, 2), -100)), dim=1)
    assert torch.allclose(example, AnswerCrossEntropyLoss(-100, smoothing, "example")(
        more_logits, more_targets))
    example.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.all(logits.grad[0, :, 1:] == 0)


def test_example_loss_rejects_answers_without_targets():
    from utils.losses import AnswerCrossEntropyLoss

    with pytest.raises(ValueError, match="non-padding"):
        AnswerCrossEntropyLoss(-100, normalization="example")(
            torch.zeros(2, 2, 3), torch.full((2, 3), -100))


def test_image_preset_does_not_override_resume_or_evaluation():
    for args in (get_args(["--dataset", "scivqa", "--resume", "old.pt"]),
                 get_args(["--dataset", "scivqa"], evaluation=True)):
        assert args.image_size is None and args.image_preprocessing is None
        assert args.freeze_question_encoder is None
    assert get_args(["--dataset", "scivqa", "--resume", "old.pt"]).loss_normalization is None
    args = get_args(["--dataset", "scivqa", "--image_size=256", "--image_preprocessing", "crop"],
                    evaluation=True)
    assert args.image_size == 256 and args.image_preprocessing == "crop"
    assert get_args(["--dataset", "scivqa", "--resume", "old.pt",
                     "--no-freeze_question_encoder"]).freeze_question_encoder is False


def test_generation_projection_matches_full_decoder_logits():
    from model.vqa_model import VQAModel
    from model.decoder_model import Decoder

    class Embeddings(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(30, 16)

        def embed_ids(self, ids):
            return self.embedding(ids)

    torch.manual_seed(1105)
    model = VQAModel.__new__(VQAModel)
    nn.Module.__init__(model)
    model.pad_token_id = 0
    model.answer_embedding = Embeddings()
    model.answer_projection = nn.Identity()
    model.decoder = Decoder(16, 32, 4, 0.2, 2)
    model.mlp = nn.Sequential(nn.Dropout(0.3), nn.Linear(16, 16), nn.GELU(), nn.Linear(16, 30))
    model.eval()
    ids = torch.tensor([[1, 5, 8, 3], [1, 4, 0, 0]])
    memory = torch.randn(2, 7, 16)
    blocked = torch.tensor([[False] * 5 + [True] * 2, [False] * 7])
    full = model.decode(ids, memory, memory_blocked=blocked)
    newest = model.decode(ids, memory, memory_blocked=blocked, last_token_only=True)
    assert newest.shape == (2, 1, 30)
    assert torch.allclose(newest[:, 0], full[:, -1], atol=1e-6)


def test_resume_cannot_silently_change_loss_objective(monkeypatch):
    import train

    monkeypatch.setattr(train, "read_checkpoint", lambda *args: {"format_version": 4})
    args = get_args(["--dataset", "scivqa", "--resume", "old.pt", "--loss_normalization", "example"])
    with pytest.raises(ValueError, match="loss_normalization does not match"):
        train._run_training(args, torch.device("cpu"), rank=0, world_size=1)


def test_resume_cannot_change_figure_context(monkeypatch):
    import train

    checkpoint = {"format_version": 4, "training_config": {"loss_normalization": "token"},
                  "model_config": {"figure_context": False}, "best_metric": {},
                  "text_model": "text", "image_model": "image"}
    monkeypatch.setattr(train, "read_checkpoint", lambda *args: checkpoint)
    args = get_args(["--dataset", "scivqa", "--resume", "old.pt", "--figure_context"])
    with pytest.raises(ValueError, match="--figure_context does not match"):
        train._run_training(args, torch.device("cpu"), rank=0, world_size=1)
