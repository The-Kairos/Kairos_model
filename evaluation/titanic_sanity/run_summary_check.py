"""Replay captured section outputs offline to isolate parsing corrections."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation/prompt_cache"))
from run_benchmark import original_module
from analyze_results import timestamp_check
from src import synopsis_synthesis as current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captured", type=Path, default=ROOT / "evaluation/prompt_cache/quality_diagnosis_20261020/revised_1")
    parser.add_argument("--source", type=Path, default=ROOT / "logs/_processed/Titanic.1997.mkv_20260312_201859.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    baseline = original_module("c32882928a4650f77717a9379acd9593a42c45bb")
    sections = json.loads((args.captured / "raw_sections.json").read_text(encoding="utf-8"))
    answers_before, answers_after = [], []
    midpoint = len(current.REQUIRED_QUESTIONS) // 2
    for suffix, required in (("a", current.REQUIRED_QUESTIONS[:midpoint]), ("b", current.REQUIRED_QUESTIONS[midpoint:])):
        rows = [r for r in sections if r["stage"] == "synopsis:qna_predefined_" + suffix]
        payload, ok = current._parse_questions_nonjson(rows[-1]["text"], len(required))
        if not ok:
            raise ValueError("Capture requires JSON/repair replay; unsupported by this check")
        questions = current._extract_questions(payload)
        answers_before.extend(baseline._normalize_predefined_questions(questions, required))
        answers_after.extend(current._normalize_predefined_questions(questions, required))
    synopsis = json.loads((args.captured / "synopsis.json").read_text(encoding="utf-8"))
    narrative = json.loads(args.source.read_bytes())["narratives"][-1]["narrative"]
    fixed = current._ground_synopsis_timestamps(synopsis, narrative)
    missing = lambda rows: sum(r["answer"] == "Not explicitly stated." for r in rows)
    report = {
        "method": "Same raw responses replayed offline through old and corrected normalizers; no new API calls.",
        "required_question_count": len(answers_before),
        "missing_answers_before": missing(answers_before),
        "missing_answers_after": missing(answers_after),
        "restored_answers": sum(a["answer"] == "Not explicitly stated." and b["answer"] != a["answer"] for a, b in zip(answers_before, answers_after)),
        "timestamps_before": timestamp_check(synopsis, narrative),
        "timestamps_after": timestamp_check(fixed, narrative),
        "limitations": ["Preserving an answer does not validate its factual accuracy.",
                        "Timestamp membership does not establish that a timestamp belongs to the stated event.",
                        "Missing source timing cannot be recovered by this guard; event times become unknown."]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "replayed_required_answers.json").write_text(json.dumps(answers_after, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
