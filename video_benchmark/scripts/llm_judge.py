"""
LLM-as-judge: score model answers against gold using a fixed rubric (Claude / Anthropic).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.utils import bench_root, load_yaml


JUDGE_SYSTEM = """You are an expert evaluator for long-video question answering benchmarks.

Your job is to compare a model's answer to a reference (gold) answer for the SAME question about the SAME video.

Scoring rubric (pick exactly one verdict):
- correct: The model answer is factually consistent with the gold answer. Minor wording differences are fine.
- partial: The model answer is on-topic and partly right but misses important details, is vague, or mixes correct and incorrect claims.
- incorrect: The model answer contradicts the gold answer, is largely wrong, or fails to address the question.
- abstain: Use only when the gold answer is unclear/empty OR the model appropriately refuses because the question cannot be answered from typical video content.

Also assess:
- hallucination_likely: true if the model adds specific claims that are not supported by or contradict the gold (and are not harmless paraphrase).

Rules:
1. Accept paraphrases and synonyms as correct when they preserve meaning.
2. If acceptable_variants are provided, treat any of them as equally valid gold.
3. Do not penalize brevity if the content is correct.
4. Output a single JSON object only, no markdown fences, no extra text.

JSON schema:
{
  "verdict": "correct|partial|incorrect|abstain",
  "confidence": <number from 0 to 1>,
  "hallucination_likely": <boolean>,
  "explanation": "<one or two short sentences>"
}
"""


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}\s*$", text)
    if m:
        text = m.group(0)
    return json.loads(text)


@dataclass
class JudgeConfig:
    model: str
    env_api_key: str
    temperature: float
    max_tokens: int


def load_judge_config(path: Path | None = None) -> JudgeConfig:
    p = path or bench_root() / "config" / "evaluation.yaml"
    raw = load_yaml(Path(p))
    lj = raw.get("llm_judge") or {}
    return JudgeConfig(
        model=lj.get("model") or "claude-sonnet-4-20250514",
        env_api_key=lj.get("env_api_key") or "ANTHROPIC_API_KEY",
        temperature=float(lj.get("temperature", 0)),
        max_tokens=int(lj.get("max_tokens", 1024)),
    )


def judge_one(
    *,
    question: str,
    gold_answer: str,
    acceptable_variants: list[str] | None,
    model_response: str,
    cfg: JudgeConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or load_judge_config()
    api_key = os.environ.get(cfg.env_api_key)
    if not api_key:
        raise RuntimeError(f"Set {cfg.env_api_key} to run the LLM judge.")

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    variants = acceptable_variants or []
    user_block = {
        "question": question,
        "gold_answer": gold_answer,
        "acceptable_variants": variants,
        "model_response": model_response,
    }
    msg = client.messages.create(
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": json.dumps(user_block, ensure_ascii=False)}],
    )
    text = ""
    for block in msg.content:
        if hasattr(block, "text"):
            text += block.text
    parsed = _extract_json_object(text)
    verdict = parsed.get("verdict", "incorrect")
    if verdict not in ("correct", "partial", "incorrect", "abstain"):
        verdict = "incorrect"
    return {
        "verdict": verdict,
        "confidence": float(parsed.get("confidence", 0)),
        "hallucination_likely": bool(parsed.get("hallucination_likely", False)),
        "explanation": str(parsed.get("explanation", "")),
        "raw_judge_text": text,
        "judge_model": cfg.model,
    }


def judge_batch(items: list[dict[str, Any]], cfg: JudgeConfig | None = None) -> list[dict[str, Any]]:
    cfg = cfg or load_judge_config()
    out = []
    for row in items:
        gold = row.get("gold") or {}
        gold_answer = gold.get("gold_answer") or gold.get("answer") or ""
        variants = gold.get("acceptable_variants")
        judgment = judge_one(
            question=row.get("question", ""),
            gold_answer=gold_answer,
            acceptable_variants=variants if isinstance(variants, list) else None,
            model_response=row.get("response", ""),
            cfg=cfg,
        )
        out.append({**row, **judgment})
    return out
