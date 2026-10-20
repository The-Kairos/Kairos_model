"""Offline checks for prompt formatting; no SDK imports or API calls."""
import ast
import json
from pathlib import Path
import string
import unittest
import os
from unittest.mock import patch
from src.prompt_layout import narrative_prefix, prepare_template, prepare_narrative_prompt

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "prompts"


def builders():
    # Load only pure builders so tests never load credentials or optional SDKs.
    source = ast.parse((ROOT / "src/synopsis_synthesis.py").read_text(encoding="utf-8"))
    names = {
        "_build_generated_fill_prompt", "_build_questions_prompt",
        "_build_safe_section_prompt", "_highlight_count_rule", "_timeline_count_rule",
    }
    module = ast.Module(body=[n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    scope = {"_narrative_prefix": narrative_prefix, "_prepare_narrative_prompt": prepare_narrative_prompt}
    exec(compile(module, "prompt_builders", "exec"), scope)
    return scope


class PromptLayoutTests(unittest.TestCase):
    def setUp(self):
        enabled = patch.dict(os.environ, KAIROS_CACHE_FRIENDLY_PROMPTS="1")
        enabled.start()
        self.addCleanup(enabled.stop)

    def test_all_templates_render_with_existing_fields(self):
        values = dict(text="SOURCE {literal braces}", carryover_context="HISTORY",
                      scene_chunk="SCENES", segment_narrative="SEGMENT", context="CONTEXT",
                      question="QUESTION", highlights_count="4-6", timeline_count="4-6",
                      clips_count="4-6", extra_questions_count=15, required_questions_count=11,
                      required_questions_block="REQUIRED", draft_json='{"summary":"draft"}')
        formatter = string.Formatter()
        for path in PROMPTS.glob("*.txt"):
            with self.subTest(template=path.name):
                template = path.read_text(encoding="utf-8")
                fields = [key for _, key, _, _ in formatter.parse(template) if key]
                self.assertTrue(set(fields) <= values.keys())
                rendered = template.format(**values)
                for key in fields:
                    self.assertIn(str(values[key]), rendered)

    def test_synopsis_tasks_share_entire_source_prefix(self):
        source = "Evidence with {braces}\n" * 600
        prefix = builders()["_narrative_prefix"](source)
        for path in list(PROMPTS.glob("synopsis_*.txt")) + [PROMPTS / "synposis_rag.txt"]:
            with self.subTest(template=path.name):
                # Only substitute source; other task fields intentionally differ.
                rendered = prepare_template(path.read_text(encoding="utf-8")).replace("{text}", source)
                self.assertTrue(rendered.startswith(prefix))
                self.assertEqual(rendered.count(source), 1)

    def test_fallback_builders_share_prefix_and_preserve_task_data(self):
        scope = builders()
        source = "Evidence with {braces}\n" * 600
        prefix = scope["_narrative_prefix"](source)
        outputs = [
            scope["_build_generated_fill_prompt"](source, ["Existing?"], 3),
            scope["_build_questions_prompt"](source, ["Required?"], 3, strict=True),
        ]
        for section in ("summary", "highlights", "timeline", "clips", "qna_predefined", "qna_generated", "qna_legacy", "monolith"):
            outputs.append(scope["_build_safe_section_prompt"](
                section, source, 4, 6, "4-6", 4, 6, "4-6", "Required?", 3))
        for output in outputs:
            self.assertTrue(output.startswith(prefix))
            self.assertEqual(output.count(source), 1)
        self.assertIn("Existing?", outputs[0])
        self.assertIn("Required?", outputs[1])

    def test_scene_and_chunk_rules_precede_changing_evidence(self):
        for name in ("describe_scene.txt", "fallback_describe_scene.txt"):
            text = prepare_template((PROMPTS / name).read_text(encoding="utf-8"))
            self.assertLess(text.index("Output format"), text.index("{{VIDEO_NAME}}"))
            self.assertLess(text.index("{{VIDEO_NAME}}"), text.index("{{SCENE_TEXT}}"))
        for name in ("chunk_summary.txt", "fallback_chunk_summary.txt"):
            text = prepare_template((PROMPTS / name).read_text(encoding="utf-8"))
            self.assertLess(text.index("Include TIMESTAMPS"), text.index("{carryover_context}"))

    def test_inline_synopsis_prompts_share_source_without_duplication(self):
        tree = ast.parse((ROOT / "src/synopsis_synthesis.py").read_text(encoding="utf-8"))
        source = "Unique narrative evidence.\n" * 600
        scope = builders()
        scope.update(narrative_text=source, highlight_min=4, highlight_max=6,
                     highlight_label="4-6", timeline_min=4, timeline_max=6,
                     timeline_label="4-6", REQUIRED_QUESTIONS=["Required?"],
                     extra_questions_count=3, required_block="Required?",
                     draft_synopsis={"summary": "draft"}, json=json)
        found = set()
        for node in sorted(ast.walk(tree), key=lambda n: getattr(n, "lineno", 0)):
            if not isinstance(node, ast.Assign):
                continue
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            selected = names & {"monolith_prompt", "consistency_prompt"}
            if selected:
                rendered = eval(compile(ast.Expression(node.value), "inline_prompt", "eval"), scope)
                scope[next(iter(selected))] = rendered
                self.assertEqual(rendered.count(source), 1)
                if isinstance(node.value, ast.Call):
                    self.assertTrue(rendered.startswith(scope["_narrative_prefix"](source)))
                    found.update(selected)
        self.assertEqual(found, {"monolith_prompt", "consistency_prompt"})


if __name__ == "__main__":
    unittest.main()
