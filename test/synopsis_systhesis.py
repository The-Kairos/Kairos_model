import os
import json
from pathlib import Path
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = 7000
FINAL_CHUNK_SIZE = CHUNK_SIZE* 5 #15000


def _debug_print(enabled: bool, message: str):
    if enabled:
        print(message)

# ----------------------
# 2. Chunk scenes
# ----------------------
def chunk_scenes(scenes: list, max_chars: int = CHUNK_SIZE, debug: bool = False):
    """
    Break scene dictionaries into <= max_chars chunks.
    """
    scene_count = len(scenes) if scenes else 0
    chunks = []
    this_chunk = ""

    for scene in scenes:
        start_timecode = scene.get("start_timecode")
        audio_speech = scene.get("audio_speech")
        llm_scene_description = scene.get("llm_scene_description")

        this_chunk += f'At {start_timecode}, {llm_scene_description}. It says "{audio_speech}".'

        if len(this_chunk) >= max_chars:
            chunks.append(this_chunk)
            this_chunk = ""

    if this_chunk:
        chunks.append(this_chunk)

    _debug_print(debug, f"chunk_scenes: {scene_count} scenes produced {len(chunks)} chunks")
    return chunks

# ----------------------
# 3. GPT call helper
# ----------------------
def call_gpt(prompt):
    """
    Minimal GPT call using AzureOpenAI client
    """
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a precise and reliable assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=4096,
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

SEGMENT_PROMPT = load_prompt("segment_prompt.txt")
CARRYOVER_PROMPT = load_prompt("carryover_prompt.txt")
SYSTHESIS_PROMPT = load_prompt("systhesis_prompt.txt")

# ----------------------
# 5. Summarize segments with carryover context
# ----------------------
def condense_chunk(chunk_text: str, pre_carryover_context: str, debug: bool = False):
    """
    Summarize a chunk and return (summary, new_carryover_context).
    """
    segment_prompt = SEGMENT_PROMPT.format(
        carryover_context=pre_carryover_context,
        scene_chunk=chunk_text
    )
    summary = call_gpt(segment_prompt)

    carryover_prompt = CARRYOVER_PROMPT.format(
        segment_narrative=summary
    )
    new_carryover_context = call_gpt(carryover_prompt)
    _debug_print(debug, f"condense_chunk: chunk_len={len(chunk_text)} condensed into len={len(summary)}")
    return summary, new_carryover_context

def chunk_narrative(narrative: str, max_chars: int = CHUNK_SIZE, debug: bool = False):
    """
    Chunk narrative into <= max_chars blocks, preferring paragraph breaks.
    """
    paragraphs = [p.strip() for p in narrative.split("\n\n") if p.strip()]
    chunks = []
    this_chunk = ""

    for para in paragraphs:
        candidate = f"{this_chunk}\n\n{para}".strip() if this_chunk else para
        if len(candidate) <= max_chars:
            this_chunk = candidate
        else:
            if this_chunk:
                chunks.append(this_chunk)
            if len(para) > max_chars:
                # Fallback: hard split a long paragraph
                for i in range(0, len(para), max_chars):
                    chunks.append(para[i:i + max_chars])
                this_chunk = ""
            else:
                this_chunk = para

    if this_chunk:
        chunks.append(this_chunk)
    _debug_print(debug, f"chunk_narrative: splitting narrative len={len(narrative)} to {len(chunks)} chunks")
    return chunks

def summarize_scenes(scene_dict, max_chars: int = FINAL_CHUNK_SIZE, debug: bool = False):
    """
    Summarize scene chunks into a narrative, then recursively compress
    until the narrative fits within max_chars.
    """
    narrative = ""
    pre_carryover_context = "None"
    debug_narratives = {}
    pass_index = 1

    scene_chunks = chunk_scenes(scene_dict, debug=debug)
    _debug_print(debug, f"summarize_scenes: round {pass_index} with {len(scene_chunks)} chunks")

    for scene in scene_chunks:
        summary, pre_carryover_context = condense_chunk(scene, pre_carryover_context, debug=debug)
        narrative += summary + "\n"

    debug_narratives[f"narrative_{pass_index}"] = {
        "len": len(narrative),
        "narrative": narrative.strip(),
        "chunks": len(scene_chunks)
    }
    (
        debug,
        f"summarize_scenes: round {pass_index} -> narrative len={len(narrative)} (chunks={len(scene_chunks)})"
    )

    while len(narrative) > max_chars:

        pass_index += 1
        narrative_chunks = chunk_narrative(narrative, max_chars=max_chars, debug=debug)
        _debug_print(debug, f"summarize_scenes: round {pass_index} with {len(narrative_chunks)} chunks")
        narrative = ""
        for chunk in narrative_chunks:
            summary, pre_carryover_context = condense_chunk(chunk, pre_carryover_context, debug=debug)
            narrative += summary + "\n"

        debug_narratives[f"narrative_{pass_index}"] = {
            "len": len(narrative),
            "narrative": narrative.strip(),
            "chunks": len(narrative_chunks)
        }
        (
            debug,
            f"summarize_scenes: round {pass_index} -> narrative len={len(narrative)} (chunks={len(narrative_chunks)})"
        )

    return narrative.strip(), debug_narratives

# ----------------------
# 6. Full narrative synthesis
# ----------------------
def synthesize_synopsis(narrative: str, debug: bool = False):
    """
    Produce a final synopsis + Q&A from the narrative.
    """
    _debug_print(debug, f"synthesize_synopsis: start (narrative len={len(narrative)})")
    return call_gpt(SYSTHESIS_PROMPT.format(text=narrative))

# ----------------------
# 7. Example usage
# ----------------------
def test(log_path):
    # scenes = load_your_scenes()  # Your list of scene dictionaries
    with open(log_path, "r", encoding="utf-8") as f:
        logs = json.load(f)

    narrative, debug_narratives = summarize_scenes(logs.get("scenes"), debug=True)
    synopsys = synthesize_synopsis(narrative, debug=True)

    synopsis_path = fr"PASTTTTAAA.txt"
    with open(synopsis_path, "w", encoding="utf-8") as f:
        f.write(synopsys)

    _debug_print(True, f"synopsis is saved in {synopsis_path}")
    print("Done! Synopsis saved.")

test(r"logs\pasta_hist3_gpt-4o_20260129_105840.json")