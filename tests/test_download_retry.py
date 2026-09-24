"""Offline checks for Hugging Face rate-limit handling and page resumption."""

import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

from utils.download_gqa import _retry_delay, fetch_bytes, fetch_json


class DownloadRetryTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
