"""Offline integration checks through the actual concurrent stage functions."""
import json
import re
import inspect
import sys
import types
import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest

from src import scene_description as scenes
from src import synopsis_synthesis as synopsis


class PipelineRegressionTests(unittest.TestCase):
    def output(self, *args, stage="", **kwargs):
        if stage == "synopsis:summary":
            return "Alice Opens Blue Box | Alice opens a blue box and finds a red ball."
        if stage == "synopsis:highlights":
            return "\n".join(f"00:00:0{i} | 00:00:0{i+1} | Alice handles the box." for i in range(1, 5))
        if stage == "synopsis:timeline":
            return "\n".join(f"00:00:0{i} | Box event {i}" for i in range(1, 5))
        if stage.startswith("synopsis:qna_predefined"):
            mid = len(synopsis.REQUIRED_QUESTIONS) // 2
            required = synopsis.REQUIRED_QUESTIONS[:mid] if stage.endswith("_a") else synopsis.REQUIRED_QUESTIONS[mid:]
            return "\n".join(re.sub(r"[^\w\s]", "", q) + " | Alice opens the blue box." for q in required)
        if stage == "synopsis:qna_generated":
            return "\n".join(f"What is object {i}? | A red ball." for i in range(15))
        if stage == "synopsis:clips":
            return json.dumps({"suggested_clips": [dict(timestamp=f"00:00:0{i}", start=f"00:00:0{i}", end=f"00:00:0{i+1}", description="Alice opens the box.") for i in range(1, 5)]})
        if stage == "synthesize_synopsis:repair:summary":
            return json.dumps(dict(chat_name="Alice Opens Blue Box", summary="Alice finds a red ball."))
        raise AssertionError("Unexpected stage: " + stage)

    def data(self):
        return {"narratives": [{"narrative": "\n".join(f"00:00:0{i}.000: Alice handles the blue box and red ball." for i in range(1, 6))}], "scenes": []}

    def test_full_synopsis_keeps_answers_times_and_saved_output(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(synopsis, "call_gpt", side_effect=self.output):
                result = synopsis.synthesize_synopsis(None, "fake", self.data(), output_dir=directory)["synopsis"]
            self.assertEqual(len(result["questions"]), 37)
            self.assertTrue(all(q["answer"] != "Not explicitly stated." for q in result["questions"][:22]))
            self.assertEqual(result["video_timeline"][0]["timestamp"], "00:00:01")
            self.assertEqual(result["suggested_clips"][0]["end"], "00:00:02")
            self.assertEqual(json.loads((Path(directory) / "synopsis.json").read_text()), result)
            self.assertTrue((Path(directory) / "synopsis.md").exists())

    def test_bad_primary_summary_is_repaired_without_losing_other_sections(self):
        def output(*args, stage="", **kwargs):
            return "malformed" if stage == "synopsis:summary" else self.output(*args, stage=stage, **kwargs)
        with patch.object(synopsis, "call_gpt", side_effect=output):
            result = synopsis.synthesize_synopsis(None, "fake", self.data())["synopsis"]
        self.assertEqual(result["summary"], "Alice finds a red ball.")
        self.assertEqual(len(result["questions"]), 37)

    def test_scene_history_uses_previous_short_summaries_in_order(self):
        captured = []
        def describe(text, client, prompt_path, **kwargs):
            if "short" in prompt_path:
                return "SHORT " + text
            captured.append(text)
            return "FINAL " + text
        inputs = [{"scene_index": i} for i in range(3)]
        with patch.object(scenes, "raw_descriptions", return_value=["A", "B", "C"]), patch.object(scenes, "describe_flash_scene", side_effect=describe):
            result = scenes.describe_scenes(inputs, None, hist_size=1, max_workers=3, cooldown_sec=0)
        self.assertEqual([r["scene_index"] for r in result], [0, 1, 2])
        b = next(text for text in captured if text.startswith("B"))
        c = next(text for text in captured if text.startswith("C"))
        self.assertIn("SHORT A", b)
        self.assertNotIn("SHORT A", c)
        self.assertIn("SHORT B", c)
        self.assertTrue(all("llm_scene_description" not in r for r in inputs))

    def test_matched_baseline_and_candidate_keep_same_pipeline_results(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation/prompt_cache"))
        from run_benchmark import original_module
        baseline = original_module("c32882928a4650f77717a9379acd9593a42c45bb", quality_matched=True)
        original_prompts, candidate_prompts = {}, {}
        def baseline_output(*args, **kwargs):
            name = inspect.currentframe().f_back.f_locals.get("name", "")
            original_prompts[name] = args[2]
            return self.output(stage="synopsis:" + name)
        def candidate_output(*args, stage="", **kwargs):
            candidate_prompts[stage.removeprefix("synopsis:")] = args[2]
            return self.output(*args, stage=stage, **kwargs)
        with patch.object(baseline, "call_gpt", new=baseline_output), patch.object(synopsis, "call_gpt", side_effect=candidate_output):
            old = baseline.synthesize_synopsis(None, "fake", self.data())["synopsis"]
            new = synopsis.synthesize_synopsis(None, "fake", self.data())["synopsis"]
        self.assertEqual(old, new)
        self.assertEqual(original_prompts, candidate_prompts)

    def test_default_layout_is_identity_and_experiment_is_opt_in(self):
        from src.prompt_layout import prepare_narrative_prompt, prepare_template
        prompt = "Task rules\nINPUT NARRATIVE:\nSource {braces}\nDRAFT JSON:\n{}"
        self.assertEqual(prepare_narrative_prompt(prompt, "Source {braces}", enabled=False), prompt)
        with patch.dict("os.environ", KAIROS_CACHE_FRIENDLY_PROMPTS=""):
            self.assertEqual(prepare_template(prompt), prompt)
        changed = prepare_narrative_prompt(prompt, "Source {braces}", enabled=True)
        self.assertEqual(changed.count("Source {braces}"), 1)
        self.assertIn("DRAFT JSON:\n{}", changed)

    def test_default_scene_api_payloads_match_original(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation/prompt_cache"))
        from run_benchmark import git_text
        baseline = types.ModuleType("original_scenes")
        exec(compile(git_text("c32882928a4650f77717a9379acd9593a42c45bb", "src/scene_description.py"), "original_scenes", "exec"), baseline.__dict__)
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="scene report"))])
        client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
        with patch.dict("os.environ", KAIROS_CACHE_FRIENDLY_PROMPTS="", KAIROS_LLM_METRICS_PATH=""):
            for name in ("describe_scene.txt", "describe_scene_short.txt", "fallback_describe_scene.txt"):
                calls.clear()
                args = dict(prompt_path=str(Path(__file__).resolve().parents[1] / "prompts" / name), model="gpt-4o", gpt_deployment="fake", video_path="test.mp4")
                baseline.describe_flash_scene("A caption with {braces}.", client, **args)
                scenes.describe_flash_scene("A caption with {braces}.", client, **args)
                self.assertEqual(calls[0], calls[1])

    def test_experiment_preserves_unrecognized_custom_scene_template(self):
        from src.prompt_layout import prepare_template
        template = "Output format: report\nVIDEO NAME: {{VIDEO_NAME}}\n{{SCENE_TEXT}}\nCustom report cue"
        self.assertEqual(prepare_template(template, enabled=True), template)


if __name__ == "__main__":
    unittest.main()
