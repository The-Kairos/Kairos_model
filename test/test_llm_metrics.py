import json
import os
from pathlib import Path
import tempfile
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.llm_metrics import measured_chat_completion, measured_stage


class MetricsTests(unittest.TestCase):
    def call(self, create, **kwargs):
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        return measured_chat_completion(client, stage="test", **kwargs)

    def test_success_preserves_response_and_omits_content(self):
        response = SimpleNamespace(model="actual-model", usage=SimpleNamespace(
            prompt_tokens=2048, completion_tokens=20,
            prompt_tokens_details=SimpleNamespace(cached_tokens=1024)),
            choices=[SimpleNamespace(finish_reason="stop")])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            with patch.dict(os.environ, KAIROS_LLM_METRICS_PATH=str(path), KAIROS_LLM_RUN_ID="run"):
                self.assertIs(self.call(lambda **kw: response, model="deployment", messages="PRIVATE TEXT"), response)
            text = path.read_text()
            self.assertNotIn("PRIVATE TEXT", text)
            row = json.loads(text)
            self.assertEqual(row["cached_tokens"], 1024)
            self.assertEqual(row["model"], "actual-model")
            self.assertEqual(row["run_id"], "run")

    def test_errors_are_reraised_without_recording_message(self):
        error = ValueError("PRIVATE ENDPOINT")
        def fail(**kwargs):
            raise error
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            with patch.dict(os.environ, KAIROS_LLM_METRICS_PATH=str(path)):
                with self.assertRaises(ValueError) as caught:
                    self.call(fail, attempt=2)
            self.assertIs(caught.exception, error)
            text = path.read_text()
            self.assertNotIn("PRIVATE ENDPOINT", text)
            self.assertEqual(json.loads(text)["attempt"], 2)

    def test_missing_usage_is_unknown_not_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            with patch.dict(os.environ, KAIROS_LLM_METRICS_PATH=str(path)):
                self.call(lambda **kw: SimpleNamespace())
            self.assertIsNone(json.loads(path.read_text())["cached_tokens"])

    def test_parallel_calls_write_valid_individual_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            with patch.dict(os.environ, KAIROS_LLM_METRICS_PATH=str(path)):
                with ThreadPoolExecutor(max_workers=8) as pool:
                    list(pool.map(lambda n: self.call(lambda **kw: SimpleNamespace()), range(40)))
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows), 40)

    def test_bad_log_path_does_not_change_api_result(self):
        response = SimpleNamespace()
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, KAIROS_LLM_METRICS_PATH=directory):
                with self.assertLogs("src.llm_metrics", level="WARNING"):
                    self.assertIs(self.call(lambda **kw: response), response)

    def test_stage_wraps_parallel_work_and_preserves_return_value(self):
        @measured_stage("parallel")
        def stage():
            with ThreadPoolExecutor(max_workers=2) as pool:
                return list(pool.map(lambda n: n * 2, [1, 2]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            with patch.dict(os.environ, KAIROS_LLM_METRICS_PATH=str(path)):
                self.assertEqual(stage(), [2, 4])
            self.assertEqual(json.loads(path.read_text())["kind"], "stage")

    def test_invalid_log_path_does_not_break_generation(self):
        result = SimpleNamespace()
        with patch("src.llm_metrics.os.getenv", side_effect=lambda key: "invalid\x00path" if key == "KAIROS_LLM_METRICS_PATH" else None):
            with self.assertLogs("src.llm_metrics", level="WARNING"):
                self.assertIs(self.call(lambda **kw: result), result)


if __name__ == "__main__":
    unittest.main()
