import os
import json
from pathlib import Path
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = 7000
FINAL_CHUNK_SIZE = CHUNK_SIZE * 5

# ----------------------
# 1. Debug helper
# ----------------------
def _debug_print(enabled: bool, message: str):
    if enabled:
        print(message)

# ----------------------
# 2. Chunk scenes
# ----------------------
def chunk_scenes(scenes: list, chunk_size: int = CHUNK_SIZE, debug: bool = False):
    """
    Break scene dictionaries into <= chunk_size chunks.
    """
    scene_count = len(scenes) if scenes else 0
    chunks = []
    this_chunk = ""

    for scene in scenes:
        start_timecode = scene.get("start_timecode")
        audio_speech = scene.get("audio_speech")
        llm_scene_description = scene.get("llm_scene_description")

        this_chunk += f'At {start_timecode}, {llm_scene_description}. It says "{audio_speech}".'

        if len(this_chunk) >= chunk_size:
            chunks.append(this_chunk)
            this_chunk = ""

    if this_chunk:
        chunks.append(this_chunk)

    _debug_print(debug, f"chunk_scenes: {scene_count} scenes turned into {len(chunks)} chunk_scenes")
    return chunks

# ----------------------
# 3. GPT call helper
# ----------------------
def call_gpt(client, deployment, prompt):
    """
    Minimal GPT call using AzureOpenAI client
    """
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a precise and reliable assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=16384,  # apparently the max for gpt4o
        temperature=1.0,
        top_p=1.0,
        model=deployment
    )
    text = response.choices[0].message.content.strip()
    return text

# ----------------------
# 4. Segment synthesis prompts (loaded from prompts folder)
# ----------------------
PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"

def load_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8")

SEGMENT_PROMPT = load_prompt("chunk_summary.txt")
CARRYOVER_PROMPT = load_prompt("chunk_summary_carryover.txt")
SYSTHESIS_PROMPT = load_prompt("synposis_rag.txt")

# ----------------------
# 5. Summarize segments with carryover context
# ----------------------
def condense_chunk(client, deployment, chunk_text: str, pre_carryover_context: str, debug: bool = False):
    """
    Summarize a chunk and return (summary, new_carryover_context).
    """
    segment_prompt = SEGMENT_PROMPT.format(
        carryover_context=pre_carryover_context,
        scene_chunk=chunk_text
    )
    summary = call_gpt(client, deployment, segment_prompt)

    carryover_prompt = CARRYOVER_PROMPT.format(
        segment_narrative=summary
    )
    new_carryover_context = call_gpt(client, deployment, carryover_prompt)
    _debug_print(debug, f"condense_chunk: chunk_len={len(chunk_text)} condensed into len={len(summary)}")
    return summary, new_carryover_context

def chunk_narrative(narrative: str, chunk_size: int = CHUNK_SIZE, debug: bool = False):
    """
    Chunk narrative into <= chunk_size blocks, preferring paragraph breaks.
    """
    paragraphs = [p.strip() for p in narrative.split("\n\n") if p.strip()]
    chunks = []
    this_chunk = ""

    for para in paragraphs:
        candidate = f"{this_chunk}\n\n{para}".strip() if this_chunk else para
        if len(candidate) <= chunk_size:
            this_chunk = candidate
        else:
            if this_chunk:
                chunks.append(this_chunk)
            if len(para) > chunk_size:
                # Fallback: hard split a long paragraph
                for i in range(0, len(para), chunk_size):
                    chunks.append(para[i:i + chunk_size])
                this_chunk = ""
            else:
                this_chunk = para

    if this_chunk:
        chunks.append(this_chunk)
    _debug_print(debug, f"chunk_narrative: splitting narrative len={len(narrative)} to {len(chunks)} chunks")
    return chunks

def summarize_scenes(client, deployment, scenes, chunk_size: int = CHUNK_SIZE, summary_len: int = FINAL_CHUNK_SIZE, debug: bool = False, output_dir: str | None = None):
    """
    Summarize scenes into a narrative, then recursively compress
    until the narrative fits within summary_len.
    """
    scene_chunks = chunk_scenes(scenes, chunk_size, debug=debug,)
    narratives = []
    pre_carryover_context = "None"

    narrative = ""
    for scene in scene_chunks:
        summary, pre_carryover_context = condense_chunk(client, deployment, scene, pre_carryover_context, debug=debug)
        narrative += summary + "\n"

    narratives.append({
        "narrative_len": len(narrative),
        "chunk_len": len(scene_chunks),
        "narrative": narrative.strip()
    })

    if debug:
        _debug_print(debug, "summarize_scenes:")
        _debug_print(
            debug,
            f"- narrative_size 1: {len(narrative)} char ({len(scene_chunks)} chunks)"
        )

    round_index = 1
    while len(narrative) > summary_len:
        round_index += 1
        narrative_chunks = chunk_narrative(narrative, chunk_size, debug=debug)
        narrative = ""
        for chunk in narrative_chunks:
            summary, pre_carryover_context = condense_chunk(client, deployment, chunk, pre_carryover_context, debug=debug)
            narrative += summary + "\n"

        narratives.append({
            "narrative_len": len(narrative),
            "chunk_len": len(narrative_chunks),
            "narrative": narrative.strip()
        })

        if debug:
            _debug_print(
                debug,
                f"- narrative_size {round_index}: {len(narrative)} char ({len(narrative_chunks)} chunks)"
            )

    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, item in enumerate(narratives, start=1):
            out_path = out_dir / f"narrative_{i}_len_{item['narrative_len']}.txt"
            out_path.write_text(item["narrative"], encoding="utf-8")
            _debug_print(debug, f"summarize_scenes: saved {out_path}")

    return {
        "scenes": scenes,
        "narratives": narratives
    }

# ----------------------
# 6. Full narrative synthesis
# ----------------------
def synthesize_synopsis(client, deployment, data: dict, debug: bool = False, output_dir: str | None = None, synopsis_ext: str = "md"):
    """
    Produce a final synopsis + Q&A from the narrative.
    """
    narratives = data.get("narratives", [])
    narrative_text = narratives[-1]["narrative"] if narratives else ""
    synopsis = call_gpt(client, deployment, SYSTHESIS_PROMPT.format(text=narrative_text))

    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ext = synopsis_ext.lstrip(".") if synopsis_ext else "md"
        synopsis_path = out_dir / f"synopsis.{ext}"
        synopsis_path.write_text(synopsis, encoding="utf-8")
        _debug_print(debug, f"synopsis is saved in {synopsis_path}")

    return {
        "scenes": data.get("scenes", []),
        "narratives": narratives,
        "synopsis": synopsis
    }

# ----------------------
# 7. Example usage
# ----------------------
def test(log_path):
    endpoint = os.getenv("GPT_ENDPOINT")
    deployment = os.getenv("GPT_DEPLOYMENT")
    subscription_key = os.getenv("GPT_KEY")
    api_version = os.getenv("GPT_VERSION")

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )
    with open(log_path, "r", encoding="utf-8") as f:
        logs = json.load(f)

    data = summarize_scenes(client, deployment, logs.get("scenes"), debug=True, output_dir="logs/synonsis_test")
    result = synthesize_synopsis(client, deployment, data, debug=True, output_dir="logs/synonsis_test")
    return result

# test(r"logs\pasta_hist3_gpt-4o_20260129_105840.json")
