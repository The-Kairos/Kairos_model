from pathlib import Path
import sys
import unittest

DIRECTORY = Path(__file__).resolve().parents[1] / "evaluation/prompt_cache"
sys.path.insert(0, str(DIRECTORY))
from run_benchmark import aggregate
from analyze_results import timestamp_check


class BenchmarkTests(unittest.TestCase):
    def test_unknown_cache_usage_does_not_become_zero(self):
        result = aggregate([dict(kind="request", elapsed_sec=1, prompt_tokens=10)])
        self.assertIsNone(result["cached_tokens"])
        self.assertIsNone(result["cached_token_fraction"])

    def test_token_hit_fraction_and_errors(self):
        rows = [dict(kind="request", elapsed_sec=1, prompt_tokens=2048, cached_tokens=1024),
                dict(kind="request", elapsed_sec=2, error_type="Timeout", attempt=2)]
        result = aggregate(rows)
        self.assertEqual(result["cached_token_fraction"], 0.5)
        self.assertEqual(result["failed_requests"], 1)
        self.assertEqual(result["retry_attempts"], 1)

    def test_timestamp_check_handles_precision_and_missing_values(self):
        result = timestamp_check({"video_timeline": [
            {"timestamp": "00:00:00"}, {"timestamp": "00:00:15"},
            {"timestamp": "Not explicitly stated"}]}, "Range 00:00:00.000 to 03:14:44.631")
        self.assertEqual(result["emitted_timestamp_fields"], 2)
        self.assertEqual(result["timestamp_fields_absent_from_narrative"], 1)


if __name__ == "__main__":
    unittest.main()
