import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation/prompt_cache"))
from run_local_pipeline import read_evidence, run_stages
from src.scene_description import format_single_description
from src.track_formatting import format_track_summaries


class LocalPipelineTests(unittest.TestCase):
    def test_saved_llm_outputs_are_not_reused(self):
        saved = {"video_path": "clip.mp4", "scenes": [{"scene_index": 0, "audio_speech": "Hello",
                 "llm_scene_description": "OLD", "llm_short_summary": "OLD SHORT"}],
                 "narratives": [{"narrative": "OLD NARRATIVE"}], "synopsis": {"summary": "OLD REPORT"}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text(json.dumps(saved))
            evidence = read_evidence(path)
        self.assertEqual(evidence, {"video_path": "clip.mp4", "scenes": [{"scene_index": 0, "audio_speech": "Hello"}]})

    def test_every_stage_receives_fresh_upstream_outputs(self):
        evidence = {"video_path": "clip.mp4", "scenes": [{"scene_index": 0}]}
        original = copy.deepcopy(evidence)
        seen = []
        def describe(scenes, client, **kwargs):
            seen.append("scene")
            scenes[0]["llm_scene_description"] = "NEW SCENE"
            self.assertEqual(kwargs["ASR_key"], "audio_speech")
            self.assertEqual(kwargs["AST_key"], "audio_natural")
            return scenes
        def summarize(client, deployment, scenes, **kwargs):
            seen.append("narrative")
            self.assertEqual(scenes[0]["llm_scene_description"], "NEW SCENE")
            return {"scenes": scenes, "narratives": [{"narrative": "NEW NARRATIVE"}]}
        def synopsis(client, deployment, data, **kwargs):
            seen.append("synopsis")
            self.assertEqual(data["narratives"][-1]["narrative"], "NEW NARRATIVE")
            self.assertEqual(data["video_path"], "clip.mp4")
            return dict(data, synopsis={"summary": "NEW FINAL"})
        scene_module = SimpleNamespace(describe_scenes=describe)
        synopsis_module = SimpleNamespace(summarize_scenes=summarize, synthesize_synopsis=synopsis)
        config = dict(scene_history=5, scene_workers=8, scene_cooldown_sec=0, deployment="fake",
                      chunk_chars=20000, target_narrative_chars=50000, narrative_workers=6, reduce_group_size=4)
        result, timing = run_stages(scene_module, synopsis_module, evidence, None, config, Path("unused"), progress=lambda *a, **k: None)
        self.assertEqual(seen, ["scene", "narrative", "synopsis"])
        self.assertEqual(evidence, original)
        self.assertEqual(result["synopsis"]["summary"], "NEW FINAL")
        self.assertGreaterEqual(timing["total_llm"], sum(timing[k] for k in ("scene_descriptions", "narrative", "synopsis")))

    def test_dropped_scenes_cannot_be_counted_as_a_speedup(self):
        config = dict(scene_history=5, scene_workers=8, scene_cooldown_sec=0)
        with self.assertRaisesRegex(RuntimeError, "dropped scenes"):
            run_stages(SimpleNamespace(describe_scenes=lambda *a, **k: []), None,
                       {"scenes": [{}], "video_path": "clip.mp4"}, None, config, Path("unused"), progress=lambda *a, **k: None)

    def test_saved_track_formatting_preserves_production_text(self):
        tracks = [dict(label="dog", track_id=2, movement="moving right, approaching",
                       start_pos="left", end_pos="center", relations=["near person #3"])]
        self.assertEqual(format_track_summaries(tracks, "narrative"), [
            "dog #2 is moving right approaching from left to center; dog #2 is near person #3"])
        self.assertEqual(format_track_summaries(tracks), [
            "dog #2: left -> center, moving right, approaching; relations: near person #3"])
        text = format_single_description(["A dog."], tracks)
        self.assertIn('Caption: "A dog."', text)
        self.assertIn("dog #2 is moving right approaching", text)

    def test_narrative_tree_reduce_preserves_output_when_triggered(self):
        from src import synopsis_synthesis as synopsis
        stages = []
        def generate(client, deployment, prompt, *, stage):
            stages.append(stage)
            return "A concise mapped scene summary." if stage == "narrative:map" else "Merged."
        scenes = [dict(scene_index=i, start_timecode=f"00:00:0{i}", end_timecode=f"00:00:0{i+1}",
                       llm_scene_description="A scene report with enough detail to form one chunk.", audio_speech="Hello.") for i in range(3)]
        with patch.object(synopsis, "call_gpt", side_effect=generate):
            result = synopsis.summarize_scenes(None, "fake", scenes, chunk_size=30, summary_len=10, max_workers=2)
        self.assertIn("narrative:reduce", stages)
        self.assertEqual(result["narratives"][-1]["narrative"], "Merged.")
        self.assertEqual(result["scenes"], scenes)

    def test_local_output_checks_flag_unavailable_source_times(self):
        from analyze_local_pipeline import check_outputs, paired_interval
        evidence = {"scenes": [{"scene_index": 1, "start_timecode": "00:00:01.000", "end_timecode": "00:00:02.000"}]}
        result = dict(scenes=[{"scene_index": 1}], synopsis={"video_timeline": [
            {"timestamp": "00:00:01", "event": "Known event"},
            {"timestamp": "00:00:05", "event": "Unsupported time"}]})
        checks = check_outputs(result, evidence)
        self.assertEqual(checks["timestamp_fields_not_exactly_in_saved_evidence"], 1)
        self.assertTrue(checks["scene_order_and_count_preserved"])
        self.assertEqual(paired_interval([1.0, 1.0, 1.0]), [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
