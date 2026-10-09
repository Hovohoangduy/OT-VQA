"""Verify zero-shot requests, scoring, and interrupted evaluation without a model."""

import base64
import csv
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import Mock, patch

from scripts.evaluate_gqa_lmstudio import (
    Example, LMStudioClient, PLANTEXPERT_SYSTEM_PROMPT, evaluate, load_examples,
    make_record, parse_args,
)
from utils.metrics import PAPER_METRICS


class LMStudioEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        (self.images / "test").mkdir(parents=True)
        # The unit tests inspect transport bytes; real decoding is checked by the live smoke run.
        (self.images / "test" / "1.jpg").write_bytes(b"test-image-one")
        (self.images / "test" / "2.jpg").write_bytes(b"test-image-two")
        self.questions = self.root / "test.csv"
        with self.questions.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["anno_id", "image", "question", "answer"])
            writer.writeheader()
            writer.writerows([
                {"anno_id": "001", "image": "test/1.jpg", "question": "What color?", "answer": "red"},
                {"anno_id": "002", "image": "test/2.jpg", "question": "Is it sunny?", "answer": "yes"},
            ])
        self.args = parse_args([
            "--questions", str(self.questions), "--images", str(self.images),
            "--output-dir", str(self.root / "results"),
        ])

    @staticmethod
    def result(text):
        return {"prediction": text, "raw_response": text, "latency_seconds": 0.5,
                "finish_reason": "stop", "usage": {}}

    def test_csv_and_official_json_keep_question_ids(self):
        self.assertEqual(load_examples(self.questions)[0].question_id, "001")
        official = self.root / "questions.json"
        official.write_text(json.dumps({
            "009": {"imageId": "1", "question": "What color?", "answer": "red"},
            "010": {"imageId": "2", "question": "What is there?"},
        }))
        examples = load_examples(official)
        self.assertEqual(examples[0], Example("009", "1.jpg", "What color?", "red"))
        self.assertIsNone(examples[1].answer)

    def test_payload_is_image_question_only_and_nonstreaming(self):
        client = LMStudioClient("http://127.0.0.1:1234", timeout=10, retries=0)
        response = {"choices": [{"message": {"content": " red\n"}, "finish_reason": "stop"}]}
        with patch.object(client, "request", return_value=response) as request:
            answer = client.answer("qwen3-vl-2b-instruct", load_examples(self.questions)[0],
                                   self.images / "test/1.jpg", 64, 42)
        endpoint, payload = request.call_args.args
        self.assertEqual(endpoint, "/chat/completions")
        self.assertEqual(client.base_url, "http://127.0.0.1:1234/v1")
        self.assertEqual([item["role"] for item in payload["messages"]], ["system", "user"])
        content = payload["messages"][1]["content"]
        self.assertEqual(content[1], {"type": "text", "text": "What color?"})
        image = content[0]["image_url"]["url"]
        self.assertTrue(image.startswith("data:image/jpeg;base64,"))
        self.assertEqual(base64.b64decode(image.split(",", 1)[1]), b"test-image-one")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(answer["prediction"], "red")
        self.assertEqual(answer["raw_response"], " red\n")

    def test_scoring_does_not_extract_answer_from_prose(self):
        example = Example("1", "1.jpg", "What color?", "red")
        exact = make_record(example, self.result("red"))
        self.assertTrue(exact["correct"])
        self.assertNotIn("vqa_accuracy", exact)
        case = make_record(example, self.result("RED"))
        self.assertFalse(case["correct"])
        self.assertTrue(case["normalized_correct"])
        for text in ["red.", "The color is red.", "Answer: red"]:
            self.assertFalse(make_record(example, self.result(text))["normalized_correct"])
        self.assertIsNone(make_record(Example("2", "2.jpg", "What?", None), self.result("yes"))["correct"])

    def test_interruption_saves_partial_outputs_and_resume_skips_successes(self):
        client = Mock(base_url="http://127.0.0.1:1234/v1")
        client.answer.side_effect = [self.result("red"), KeyboardInterrupt()]
        with patch("builtins.print"), self.assertRaises(KeyboardInterrupt):
            evaluate(self.args, client)
        output = self.args.output_dir
        partial = json.loads((output / "report.json").read_text())
        self.assertEqual(partial["examples_attempted"], 1)
        self.assertFalse(partial["complete"])
        self.args.resume = True
        client.answer = Mock(return_value=self.result("no"))
        with patch("builtins.print"):
            report = evaluate(self.args, client)
        self.assertEqual(client.answer.call_count, 1)
        self.assertEqual(client.answer.call_args.args[1].question_id, "002")
        self.assertTrue(report["complete"])
        self.assertEqual(report["accuracy"], 0.5)
        self.assertNotIn("vqa_accuracy", report)
        predictions = json.loads((output / "gqa_predictions.json").read_text())
        self.assertEqual(predictions, [{"questionId": "001", "prediction": "red"},
                                      {"questionId": "002", "prediction": "no"}])
        with (output / "predictions.csv").open(newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 2)
        self.args.max_tokens = 128
        with self.assertRaisesRegex(ValueError, "Resume configuration differs"):
            evaluate(self.args, client)

    def test_failed_requests_count_as_wrong_and_are_retried_on_resume(self):
        self.args.continue_on_error = True
        client = Mock(base_url="http://127.0.0.1:1234/v1")
        client.answer.side_effect = [RuntimeError("server error"), self.result("yes")]
        with patch("builtins.print"):
            report = evaluate(self.args, client)
        self.assertEqual(report["accuracy"], 0.5)
        self.assertEqual(report["errors"], 1)
        self.assertFalse(report["complete"])
        self.args.resume = True
        client.answer = Mock(return_value=self.result("red"))
        with patch("builtins.print"):
            report = evaluate(self.args, client)
        self.assertEqual(client.answer.call_count, 1)
        self.assertEqual(report["accuracy"], 1)
        self.assertEqual(report["errors"], 0)
        self.assertTrue(report["complete"])

    def test_missing_images_stop_before_any_inference(self):
        self.args.images = self.root / "missing"
        client = Mock(base_url="http://127.0.0.1:1234/v1")
        with self.assertRaises(FileNotFoundError):
            evaluate(self.args, client)
        client.answer.assert_not_called()
        self.assertFalse(self.args.output_dir.exists())

    def test_http_transient_retry_and_permanent_failure(self):
        client = LMStudioClient("http://localhost:1234/v1", 10, 1)
        transient = urllib.error.HTTPError("local", 503, "busy", {}, io.BytesIO(b"busy"))
        response = io.BytesIO(b'{"data": [{"id": "qwen3-vl-2b-instruct"}]}')
        with patch("urllib.request.urlopen", side_effect=[transient, response]) as request, \
                patch("time.sleep") as sleep:
            self.assertEqual(client.models(), ["qwen3-vl-2b-instruct"])
        self.assertEqual(request.call_count, 2)
        sleep.assert_called_once()
        permanent = urllib.error.HTTPError("local", 400, "bad", {}, io.BytesIO(b"vision unavailable"))
        with patch("urllib.request.urlopen", side_effect=permanent) as request, \
                self.assertRaisesRegex(RuntimeError, "HTTP 400: vision unavailable"):
            client.models()
        self.assertEqual(request.call_count, 1)

    def plant_args(self, *flags):
        return parse_args([
            "--dataset", "plantexpert", "--questions", str(self.questions),
            "--images", str(self.images), "--output-dir", str(self.root / "plant_results"),
            *flags,
        ])

    def test_dataset_defaults_and_overrides(self):
        gqa = parse_args([])
        self.assertEqual(gqa.max_tokens, 64)
        self.assertEqual(gqa.questions, Path("data/gqa_dataset/test.csv"))
        plant = parse_args(["--dataset", "plantexpert"])
        self.assertEqual(plant.questions, Path("data/plantexpert_dataset/test.csv"))
        self.assertEqual(plant.images, Path("data/plantexpert_dataset/images"))
        self.assertEqual(plant.output_dir, Path("results/qwen3_vl_2b_plantexpert_zero_shot"))
        self.assertEqual(plant.max_tokens, 256)
        self.assertFalse(plant.no_bertscore)
        self.assertEqual(self.plant_args("--max-tokens", "512").max_tokens, 512)

    def test_plant_request_has_no_reference_or_annotation_leakage(self):
        client = LMStudioClient("http://localhost:1234", 10, 0)
        example = Example("id", "test/1.jpg", "What disease?", "secret reference",
                          {"crop": "secret crop", "disease": "secret disease"})
        response = {"choices": [{"message": {"content": "Leaf spot is visible."}}]}
        with patch.object(client, "request", return_value=response) as request:
            client.answer("vision", example, self.images / example.image, 256, 42,
                          system_prompt=PLANTEXPERT_SYSTEM_PROMPT)
        payload = request.call_args.args[1]
        self.assertEqual(payload["messages"][0]["content"], PLANTEXPERT_SYSTEM_PROMPT)
        self.assertEqual(payload["max_tokens"], 256)
        self.assertNotIn("secret", json.dumps(payload))
        self.assertEqual(payload["messages"][1]["content"][1]["text"], example.question)

    def test_plant_lexical_metrics_and_metadata_without_dependencies(self):
        # Plant annotations are kept for analysis and never enter the request.
        with self.questions.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "anno_id", "image", "question", "answer", "question_category", "crop", "disease",
            ])
            writer.writeheader()
            writer.writerows([
                {"anno_id": "001", "image": "test/1.jpg", "question": "What disease?",
                 "answer": "Leaf spot", "question_category": "Identification",
                 "crop": "tomato", "disease": "leaf spot"},
                {"anno_id": "002", "image": "test/2.jpg", "question": "Is it healthy?", "answer": "no"},
            ])
        args = self.plant_args("--no-bertscore")
        client = Mock(base_url="http://localhost:1234/v1")
        client.answer.side_effect = [self.result("leaf spot"), self.result("yes")]
        with patch("builtins.print"), \
                patch("scripts.evaluate_gqa_lmstudio.build_bertscore_scorer") as build:
            report = evaluate(args, client)
        build.assert_not_called()
        self.assertEqual(report["generated_metrics"]["em"], 0.5)
        self.assertEqual(report["generated_metrics"]["bleu_2"], 0.5)
        self.assertIsNone(report["generated_metrics"]["bertscore_f1"])
        self.assertFalse(report["bertscore"]["enabled"])
        self.assertNotIn("accuracy", report)
        self.assertFalse((args.output_dir / "gqa_predictions.json").exists())
        with (args.output_dir / "predictions.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["question_category"], "Identification")
        self.assertEqual(rows[0]["em"], "1.0")
        self.assertEqual(rows[0]["bertscore_f1"], "")
        self.assertEqual(client.answer.call_args.kwargs["system_prompt"], PLANTEXPERT_SYSTEM_PROMPT)

    def test_plant_bertscore_failures_and_resume(self):
        args = self.plant_args("--continue-on-error")
        client = Mock(base_url="http://localhost:1234/v1")
        client.answer.side_effect = [self.result("red"), RuntimeError("server failed")]
        scorer = Mock(hash="test-encoder-hash")
        scorer.score.return_value = (None, None, [0.8])
        with patch("builtins.print"), patch(
            "scripts.evaluate_gqa_lmstudio.build_bertscore_scorer", return_value=scorer,
        ) as build:
            report = evaluate(args, client)
        build.assert_called_once_with(model_type="bert-base-uncased", device="cpu",
                                      batch_size=16, rescale_with_baseline=True)
        scorer.score.assert_called_once_with(["red"], ["red"])
        self.assertEqual(report["generated_metrics"]["bertscore_f1"], 0.4)
        self.assertEqual(report["generated_metrics"]["em"], 0.5)
        self.assertEqual(report["bertscore"]["model_hash"], "test-encoder-hash")
        self.assertFalse(report["complete"])
        args.resume = True
        client.answer = Mock(return_value=self.result("yes"))
        scorer.score.return_value = (None, None, [0.8, 0.9])
        with patch("builtins.print"), patch(
            "scripts.evaluate_gqa_lmstudio.build_bertscore_scorer", return_value=scorer,
        ):
            report = evaluate(args, client)
        self.assertEqual(client.answer.call_count, 1)
        self.assertTrue(report["complete"])
        self.assertAlmostEqual(report["generated_metrics"]["bertscore_f1"], 0.85)
        args.no_bertscore = True
        with self.assertRaisesRegex(ValueError, "Resume configuration differs"):
            evaluate(args, client)

    def test_plant_unlabeled_questions_do_not_load_bertscore(self):
        with self.questions.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image", "question"])
            writer.writeheader()
            writer.writerow({"image": "test/1.jpg", "question": "What disease?"})
        args = self.plant_args()
        client = Mock(base_url="http://localhost:1234/v1")
        client.answer.return_value = self.result("Leaf spot")
        with patch("builtins.print"), \
                patch("scripts.evaluate_gqa_lmstudio.build_bertscore_scorer") as build:
            report = evaluate(args, client)
        build.assert_not_called()
        self.assertTrue(report["complete"])
        self.assertTrue(all(value is None for value in report["generated_metrics"].values()))

    def test_plant_missing_scorer_stops_before_inference(self):
        args = self.plant_args()
        client = Mock(base_url="http://localhost:1234/v1")
        with patch("builtins.print"), patch(
            "scripts.evaluate_gqa_lmstudio.build_bertscore_scorer",
            side_effect=RuntimeError("BERTScore unavailable"),
        ), self.assertRaisesRegex(RuntimeError, "BERTScore unavailable"):
            evaluate(args, client)
        client.answer.assert_not_called()
        self.assertFalse((args.output_dir / "run_config.json").exists())

    def test_plant_empty_reference_and_failed_requests(self):
        example = Example("id", "1.jpg", "What?", "")
        record = make_record(example, error="failed", dataset="plantexpert")
        self.assertTrue(all(record[metric] == 0 for metric in PAPER_METRICS[:-1]))
        unlabeled = make_record(Example("id", "1.jpg", "What?", None),
                                self.result("leaf spot"), dataset="plantexpert")
        self.assertTrue(all(unlabeled[metric] is None for metric in PAPER_METRICS[:-1]))


if __name__ == "__main__":
    unittest.main()
