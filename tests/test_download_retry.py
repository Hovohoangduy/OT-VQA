"""Offline checks for Hugging Face rate-limit handling and page resumption."""

import tempfile
import io
import unittest
import urllib.error
from email.utils import formatdate
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image

from utils.download_gqa import (
    _pause_metadata_requests, _retry_delay, _wait_for_metadata_slot,
    download_image, fetch_bytes, fetch_json,
)


class DownloadRetryTests(unittest.TestCase):
    def test_retry_after_http_date_and_larger_reset_are_honored(self):
        limited = urllib.error.HTTPError(
            "https://datasets-server.huggingface.co/rows", 429,
            "Too Many Requests", {"Retry-After": formatdate(120, usegmt=True),
                                  "RateLimit": '"api";r=0;t=400'}, None,
        )
        with patch("utils.download_gqa.time.time", return_value=100):
            self.assertEqual(_retry_delay(limited, 0), 401)
        limited.headers = {"Retry-After": formatdate(120, usegmt=True)}
        with patch("utils.download_gqa.time.time", return_value=100):
            self.assertEqual(_retry_delay(limited, 0), 21)

    def test_invalid_retry_header_uses_backoff(self):
        for header in ("invalid", "NaN", "inf"):
            limited = urllib.error.HTTPError(
                "url", 429, "Too Many Requests", {"Retry-After": header}, None
            )
            self.assertEqual(_retry_delay(limited, 2), 20)

    def test_spacing_and_shared_cooldown_with_fake_clock(self):
        now = [100.0]
        sleeps = []

        def sleep(delay):
            sleeps.append(delay)
            now[0] += delay

        with patch("utils.download_gqa._next_metadata_request", 0.0), \
             patch("utils.download_gqa._request_interval", 2.0), \
             patch("utils.download_gqa.time.monotonic", side_effect=lambda: now[0]), \
             patch("utils.download_gqa.time.sleep", side_effect=sleep):
            _wait_for_metadata_slot()
            _wait_for_metadata_slot()
            self.assertEqual(sleeps, [2.0])
            _pause_metadata_requests(65)
            _pause_metadata_requests(3)  # A shorter cooldown must not shorten it.
            _wait_for_metadata_slot()
            self.assertEqual(sleeps, [2.0, 60, 5.0])

    def test_rate_limit_reset_header(self):
        limited = urllib.error.HTTPError(
            "https://datasets-server.huggingface.co/rows", 429,
            "Too Many Requests", {"RateLimit": '"api";r=0;t=41'}, None,
        )
        self.assertEqual(_retry_delay(limited, 0), 42.0)

    def test_429_uses_retry_after_and_retries(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"rows": []}'
        limited = urllib.error.HTTPError(
            "https://datasets-server.huggingface.co/rows", 429,
            "Too Many Requests", {"Retry-After": "3"}, None,
        )
        with patch("utils.download_gqa.urllib.request.urlopen",
                   side_effect=[limited, response]) as open_url, \
             patch("utils.download_gqa._wait_for_metadata_slot"), \
             patch("utils.download_gqa._pause_metadata_requests") as pause:
            result = fetch_bytes("https://datasets-server.huggingface.co/rows", attempts=2)
        self.assertEqual(result, b'{"rows": []}')
        self.assertEqual(open_url.call_count, 2)
        pause.assert_called_once_with(4.0)

    def test_404_is_not_retried(self):
        missing = urllib.error.HTTPError(
            "https://datasets-server.huggingface.co/rows", 404,
            "Not Found", {}, None,
        )
        with patch("utils.download_gqa.urllib.request.urlopen", side_effect=missing) as open_url, \
             patch("utils.download_gqa._wait_for_metadata_slot"):
            with self.assertRaises(urllib.error.HTTPError):
                fetch_bytes("https://datasets-server.huggingface.co/rows")
        self.assertEqual(open_url.call_count, 1)

    def test_cached_rows_are_reused_on_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            url = "https://datasets-server.huggingface.co/rows?offset=0"
            with patch("utils.download_gqa.fetch_bytes", return_value=b'{"rows": [1]}') as fetch:
                first = fetch_json(url, Path(directory))
                second = fetch_json(url, Path(directory))
            self.assertEqual(first, second)
            self.assertEqual(fetch.call_count, 1)
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 1)

    def test_corrupt_cache_is_refetched(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("utils.download_gqa.fetch_bytes", return_value=b'{"rows": [1]}') as fetch:
                fetch_json("https://datasets-server.huggingface.co/rows", Path(directory))
                next(Path(directory).glob("*.json")).write_text("{truncated")
                payload = fetch_json("https://datasets-server.huggingface.co/rows", Path(directory))
            self.assertEqual(payload, {"rows": [1]})
            self.assertEqual(fetch.call_count, 2)
            self.assertEqual(list(Path(directory).glob("*.part")), [])

    def test_cached_expired_image_url_is_refreshed(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("utils.download_gqa.fetch_bytes", side_effect=[
                b'{"rows": [{"row": {"image": {"src": "https://image?Expires=1"}}}]}',
                b'{"rows": [{"row": {"image": {"src": "https://image?Expires=9999999999"}}}]}',
            ]) as fetch:
                fetch_json("https://datasets-server.huggingface.co/rows", Path(directory))
                payload = fetch_json("https://datasets-server.huggingface.co/rows", Path(directory))
                self.assertEqual(fetch_json("https://datasets-server.huggingface.co/rows", Path(directory)), payload)
            self.assertEqual(fetch.call_count, 2)
            self.assertEqual(payload["rows"][0]["row"]["image"]["src"], "https://image?Expires=9999999999")

    def test_image_download_refreshes_evicted_viewer_asset(self):
        content = io.BytesIO()
        Image.new("RGB", (2, 2)).save(content, format="JPEG")
        missing = urllib.error.HTTPError("image", 404, "Not Found", {}, None)
        record = {"id": "a", "url": "https://image/old", "row_url": "https://metadata/row"}
        with tempfile.TemporaryDirectory() as directory:
            with patch("utils.download_gqa.fetch_bytes", side_effect=[missing, content.getvalue()]) as fetch, \
                 patch("utils.download_gqa.fetch_json", return_value={
                     "rows": [{"row": {"image": {"src": "https://image/new"}}}]
                 }) as metadata:
                self.assertEqual(download_image(record, Path(directory)), "a.jpg")
                metadata.assert_called_once_with(record["row_url"])
                self.assertEqual(fetch.call_args.args, ("https://image/new",))
                download_image(record, Path(directory))
                self.assertEqual(fetch.call_count, 2)  # Existing images are reused.

    def test_token_is_only_sent_to_metadata_hosts(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b"image"
        with patch.dict("os.environ", {"HF_TOKEN": "test-token"}), \
             patch("utils.download_gqa._wait_for_metadata_slot"), \
             patch("utils.download_gqa.urllib.request.urlopen", return_value=response) as open_url:
            fetch_bytes("https://datasets-server.huggingface.co/rows")
            self.assertEqual(open_url.call_args.args[0].get_header("Authorization"), "Bearer test-token")
            fetch_bytes("https://external.example/image.jpg")
            self.assertIsNone(open_url.call_args.args[0].get_header("Authorization"))
            fetch_bytes("https://datasets-server.huggingface.co/cached-assets/image.jpg")
            self.assertIsNone(open_url.call_args.args[0].get_header("Authorization"))


if __name__ == "__main__":
    unittest.main()
