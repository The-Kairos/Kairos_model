import unittest

from src.synopsis_synthesis import (
    _ground_synopsis_timestamps, _normalize_predefined_questions,
    _normalize_generated_questions,
)


class SynopsisQualityTests(unittest.TestCase):
    def test_question_punctuation_does_not_drop_answer(self):
        required = ["What is happening in the video?", "What context is missing or assumed? What would require outside knowledge?"]
        supplied = [dict(question="What is happening in the video", answer="An expedition explores the Titanic."),
                    dict(question="What context is missing or assumed What would require outside knowledge", answer="Historical context.")]
        self.assertEqual(_normalize_predefined_questions(supplied, required), [
            dict(question=required[0], answer=supplied[0]["answer"]),
            dict(question=required[1], answer=supplied[1]["answer"])])

    def test_different_question_is_not_matched_by_position(self):
        result = _normalize_predefined_questions([dict(question="Who is Jack?", answer="Jack.")], ["Who is Rose?"])
        self.assertEqual(result[0]["answer"], "Not explicitly stated.")

    def test_required_question_without_punctuation_is_not_generated_extra(self):
        result = _normalize_generated_questions([dict(question="Who is Rose", answer="A survivor.")], ["Who is Rose?"], 1, pad=False)
        self.assertEqual(result, [])

    def test_overall_range_does_not_support_event_time(self):
        original = {"video_timeline": [{"timestamp": "00:00:00", "event": "Opening"}],
                    "suggested_clips": [{"start": "00:00:15", "end": "03:14:44.631", "timestamp": "00:00:15"}]}
        result = _ground_synopsis_timestamps(original, "### Summary (00:00:00.000 to 03:14:44.631)\nA story.")
        self.assertEqual(result["video_timeline"][0]["timestamp"], "Not explicitly stated")
        self.assertEqual(result["suggested_clips"][0]["end"], "Not explicitly stated")
        self.assertEqual(original["video_timeline"][0]["timestamp"], "00:00:00")

    def test_explicit_source_times_preserved_across_precision(self):
        source = "00:00:15.000: Jack boards.\n00:00:30.500: Rose arrives."
        result = _ground_synopsis_timestamps({"video_highlights": [
            {"start": "00:00:15", "end": "00:00:30.500", "highlight": "Boarding"}],
            "video_timeline": [{"timestamp": "00:00:45", "event": "Invented time"}]}, source)
        self.assertEqual(result["video_highlights"][0]["start"], "00:00:15")
        self.assertEqual(result["video_highlights"][0]["end"], "00:00:30.500")
        self.assertEqual(result["video_timeline"][0]["timestamp"], "Not explicitly stated")

    def test_scene_heading_range_remains_usable(self):
        source = "### Scene 3 (00:00:15.000 to 00:00:30.500)\nJack boards."
        result = _ground_synopsis_timestamps({"suggested_clips": [
            {"start": "00:00:15", "end": "00:00:30.500"}]}, source)
        self.assertEqual(result["suggested_clips"][0]["start"], "00:00:15")
        self.assertEqual(result["suggested_clips"][0]["end"], "00:00:30.500")


if __name__ == "__main__":
    unittest.main()
