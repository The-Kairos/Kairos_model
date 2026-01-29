import os
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()

# ----------------------
# 1. Create the AzureOpenAI client
# ----------------------

endpoint = "https://60099-m1xc2jq0-australiaeast.openai.azure.com/"
model_name = "gpt-4o"
deployment = os.getenv("GPT_DEPLOYMENT")

subscription_key = os.getenv("SUPSCRIPTION_KEY")
api_version = os.getenv("API_VERSION")

from openai import AzureOpenAI
client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

# ----------------------
# 2. Chunk scenes
# ----------------------
def format_story_chunks(scenes: list, max_chars: int = 3000):
    """
    Break scenes into chunks for segment-level synthesis
    """
    chunks = []
    this_chunk = ""
    start_index = None
    start_time = None

    for scene in scenes:
        scene_index = scene.get("scene_index")
        start_timecode = scene.get("start_timecode")
        end_timecode = scene.get("end_timecode")
        audio_speech = scene.get("audio_speech")
        llm_scene_description = scene.get("llm_scene_description")

        if start_index is None:
            start_index = scene_index
            start_time = start_timecode

        this_chunk += f'At {start_timecode}, {llm_scene_description}. It says "{audio_speech}".'

        if len(this_chunk) >= max_chars:
            chunks.append((start_index, scene_index, start_time, end_timecode, this_chunk))
            this_chunk = ""
            start_index = None
            start_time = None

    if this_chunk:
        chunks.append((start_index, scene_index, start_time, end_timecode, this_chunk))

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
    return response.choices[0].message.content.strip()

# ----------------------
# 4. Segment synthesis prompts
# ----------------------
SEGMENT_PROMPT = """
You are a factual narrative generator. Your task is to write a **story-like, readable synopsis** of a video segment.

Rules:
- Write in **chronological order**, following the scenes as they happen.
- Include only **essential visual elements, actions, and spoken/audio cues** that are relevant to the story.
- Do **not** quote dialogue.
- Do **not** mention minor objects or incidental details that do not affect the story flow.
- Do **not** add interpretation, opinions, emotions, or inferred intent.
- Maintain **continuity** with previous segments using the carryover context.
- Write as if the narrative is being read as a continuous story, not a list of facts.
- Write timestaps as references

Carryover context from previous segments:
{carryover_context}

Scenes for this segment:
{scene_chunk}

Write a flowing narrative paragraph (or multiple paragraphs if needed) for this segment.
"""

CARRYOVER_PROMPT = """
Extract carryover context for the next segment.

From the narrative below, list:
- Active characters
- Unresolved objects or events
- Current setting

Rules:
- Be concise
- Use bullet points
- Do NOT infer beyond the text

Narrative:
{segment_narrative}
"""

# ----------------------
# 5. Generate segment-level narratives
# ----------------------
import json
def generate_segment_narratives(chunks):
    """
    chunks: list of tuples
      (start_index, end_index, start_time, end_time, chunk_text)
    """
    results = []
    carryover_context = "None"

    for i, (start_idx, end_idx, start_tc, end_tc, chunk_text) in enumerate(chunks):
        print(f"Processing segment {i+1}/{len(chunks)}")

        # 1. Generate segment narrative
        segment_prompt = SEGMENT_PROMPT.format(
            carryover_context=carryover_context,
            scene_chunk=chunk_text
        )

        segment_narrative = call_gpt(segment_prompt)
        print('segment_narrative', segment_narrative , "\n\n")
        # 2. Extract carryover context
        carryover_prompt = CARRYOVER_PROMPT.format(
            segment_narrative=segment_narrative
        )
        new_carryover_context = call_gpt(carryover_prompt)
        print('new_carryover_context', new_carryover_context , "\n\n")
        # 3. Store segment
        results.append({
            "segment_id": i,
            "scene_index_start": start_idx,
            "scene_index_end": end_idx,
            "time_range": f"{start_tc}–{end_tc}",
            "segment_narrative": segment_narrative,
            "carryover_context": new_carryover_context
        })

        # 4. Update carryover for next segment
        carryover_context = new_carryover_context

    return results

# ----------------------
# 6. Optional: Full narrative synthesis
# ----------------------
def synthesize_full_narrative(segments):
    """
    Combine all segment narratives into a single chronological story
    """
    text = "\n\n".join(s["segment_narrative"] for s in segments)
    prompt = f"""
Combine the following narrative segments into a single continuous factual account.
Maintain chronological order.
Do not omit events.

{text}
"""
    full_narrative = call_gpt(prompt)
    return full_narrative

# ----------------------
# 7. Example usage
# ----------------------
if __name__ == "__main__":
    # scenes = load_your_scenes()  # Your list of scene dictionaries
    log_path= r"logs\car_hist3_gpt-4o_20260123_221906.json"
    with open(log_path, "r", encoding="utf-8") as f:
        logs = json.load(f)

    chunks = format_story_chunks(logs.get("scenes"))
    segments = generate_segment_narratives(chunks)
    full_narrative = synthesize_full_narrative(segments)

    # Save results for RAG
    with open(r"logs\segment_narratives_2.json", "w") as f:
        json.dump(segments, f, indent=2)

    with open(r"logs\full_narrative_2.txt", "w") as f:
        f.write(full_narrative)

    print("Done! Segment narratives and full narrative saved.")
