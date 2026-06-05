# Kairos Knowledge Graph

This document explains how the Kairos knowledge graph was designed, why those decisions were made, and how it is currently created and exported.

## Goal

The goal of the knowledge graph is to capture structured scene-level facts from the video pipeline without changing the synopsis, RAG, or narrative workflow.

The graph is built per video, not globally across videos.

That decision affected several parts of the design:

- `Video` is not used as a graph node category
- the checkpoint stores one graph per processed video
- Neo4j uses one database per video
- reruns rebuild the same video graph from scratch

## Where The KG Fits In The Pipeline

The knowledge graph is built during `kg_extract`, after `llm_scene_description` exists.

High-level order:

1. Scene detection
2. Visual and audio analysis
3. LLM scene descriptions
4. `kg_extract`
5. Narrative summary
6. Synopsis
7. Embeddings

The graph is intentionally separate from synopsis generation. It exists as its own artifact and should not affect the narrative outputs.

## Why KG Extraction Happens After Long Scene Descriptions

One of the main design decisions was whether graph extraction should replace the second phase of scene descriptions or happen after it.

The chosen design was:

- keep the existing two-phase scene description system
- use short scene summaries to build a global node inventory
- use the long scene descriptions to extract scene-level relationships

This was chosen because the long scene descriptions are already the best fused representation of the scene. They combine:

- frame captions
- object detections
- audio speech
- natural sound
- prior scene context

They also already have graph-friendly structure:

- `Characters`
- `Key Dialogue`
- `Causal Relationships`
- `Objects and Setting`
- `Cross-Scene Continuity`

That makes them much better for relationship extraction than raw captions or short summaries.

## Node Categories

The final fixed node categories are:

- `Scene`
- `Character`
- `Object`
- `Location`
- `Action`
- `Emotion`
- `Topic`

Important decisions:

- `Video` was removed because the graph is already one graph per video
- `Event` was rejected because it was too vague and would become a catch-all bucket
- `Utterance` was discussed but deferred because it added complexity without being necessary for the first version

`Scene` is not part of the normalized node inventory, but it is still created later as a real node in Neo4j.

## Relationship Categories

The current supported directed relationships are:

- `Scene <HAS_TOPIC> Topic`
- `Character/Object/Location <IS_SHOWN_IN> Scene`
- `Character/Object <IS_IN> Location`
- `Action <OCCURS_IN> Scene`
- `Character <DOES> Action`
- `Character <INTERACTS_WITH> Object`
- `Action <INVOLVES> Object`
- `Action <TARGETS> Character/Object`
- `Character <FEELS> Emotion`
- `Action/Character/Object/Topic <CAUSES> Emotion`
- `Character <SPEAKS_TO> Character`
- `Character <MENTIONS> Topic/Character/Object/Location`
- `Scene <NEXT> Scene`

All relationships are treated as unidirectional.

One important design choice was to keep the saved relationship JSON minimal:

```json
{
  "type": "FEELS",
  "source_id": "char:sheldon",
  "target_id": "emotion:nervousness"
}
```

The extra fields were intentionally removed:

- no `source_label`
- no `target_label`
- no `direction`

This was done because:

- labels can be recovered from the node ids
- direction is already implied by the record shape
- a smaller JSON format is easier to validate and export

## Step 1: Build The Video-Level Node Inventory

Before scene relationship extraction, Kairos builds a normalized node list for the whole video.

This happens from the short scene summaries generated during the first phase of `describe_scenes()`.

The output is stored in the checkpoint like this:

```json
"knowledge_graph": {
  "nodes": {
    "Character": [...],
    "Object": [...],
    "Location": [...],
    "Action": [...],
    "Emotion": [...],
    "Topic": [...]
  }
}
```

This node inventory exists to make later extraction more consistent. Instead of letting the relationship stage invent labels independently per scene, it must work from a shared video-level vocabulary.

This logic lives in:

- `src/kg_node_list.py`
- `prompts/kg_node_list.txt`

## Step 2: Extract Scene Relationships

For each scene, Kairos uses:

- `llm_scene_description`
- the normalized video-level node inventory

to extract scene-local relationships.

### Original approach and why it failed

The first design asked the LLM to directly produce canonical JSON-ready ids like:

```json
{
  "type": "FEELS",
  "source_id": "char:sheldon",
  "target_id": "emotion:nervousness"
}
```

That failed in practice because it was too strict. The model had to:

- understand the scene
- choose the correct relation type
- choose the correct node
- normalize the label
- convert it into the exact canonical id

Any mismatch caused the validator to drop the edge.

### Current approach

The extractor now asks the LLM for something closer to natural speech:

```text
Character Sheldon <IS_SHOWN_IN> Scene this_scene
Character Sheldon <FEELS> Emotion nervousness
```

Python then parses those lines and resolves the labels to canonical ids locally.

This was chosen because it is much easier for the model to do reliably.

### Scene placeholder

The LLM uses `Scene this_scene` in its output.

Python replaces that with the real scene id when converting the line into JSON.

### Error handling

Another design decision was to avoid silent failures.

If relationship parsing fails, the scene now stores an explicit error marker rather than silently saving an empty list:

```json
"relationships": [
  "ERROR: <raw llm output or exception>"
]
```

This makes the extraction stage debuggable.

This logic lives in:

- `src/kg_node_list.py`
- `prompts/kg_relationships.txt`

## Why `relationships: []` Happened During Debugging

While debugging the extractor, three main issues were found:

1. The OpenAI path was still forcing `response_format={"type": "json_object"}` even after the prompt changed to line-based output.
2. Empty model output was being treated as a valid no-relationship result.
3. The fallback only stored `ERROR:` when raw output was non-empty, so empty responses still became plain `[]`.

Those were fixed by:

- removing JSON forcing for relationship extraction
- treating empty output as an error
- always writing an explicit `ERROR:` marker if parsing fails

## Scene Nodes In Neo4j

Each scene becomes a `:Scene` node with these properties:

- `scene_index`
- `start_timecode`
- `end_timecode`
- `frame_captions`
- `yolo_detections`
- `audio_speech`
- `audio_natural`
- `llm_scene_description`

`Scene <NEXT> Scene` is enforced for every consecutive scene pair.

One technical Neo4j constraint affected the design:

- Neo4j properties can only store primitive values or arrays of primitives
- `yolo_detections` is a list of maps

So `yolo_detections` is stored in Neo4j as a JSON string while keeping the same property name.

## Relationship Provenance

A scene cannot be stored as a node property inside a Neo4j relationship. Neo4j relationship properties are only primitive values, not node references.

The chosen approach was:

- keep the saved checkpoint relationship minimal
- add `scene_id` as a Neo4j relationship property during export

Example in Neo4j:

```cypher
(a)-[:FEELS {scene_id: 'scene:12'}]->(b)
```

This keeps provenance without complicating the stored checkpoint schema.

## Neo4j Export Design

Neo4j export is the second half of `kg_extract`.

It does three things:

1. Create a valid database name from the video filename
2. Recreate the database from scratch
3. Load scene nodes, category nodes, `NEXT` edges, and extracted scene relationships

### Database per video

One major design decision was to use one Neo4j database per video.

That means:

- the database name is derived from the video name
- all nodes and relationships for that video live in one isolated Neo4j database

Example:

```text
Young Sheldon - First Day of High School.mp4
```

becomes:

```text
young-sheldon-first-day-of-high-school
```

### Rerun strategy

Another important design choice was how reruns should behave.

The selected behavior is:

- do not append
- do not attempt incremental merge
- recreate the whole video database from scratch

This uses:

```cypher
CREATE OR REPLACE DATABASE <db_name>
```

This was chosen because the graph is per-video and rebuilding it avoids duplicates cleanly.

### Connection details

The current local Neo4j connection is:

- URL: `neo4j://127.0.0.1:7687`
- Username: `neo4j`
- Password: `kairos_kg`

The export code currently supports the local Kairos KG instance and stores export metadata back into the checkpoint under:

```json
"knowledge_graph": {
  "neo4j": {
    ...
  }
}
```

## Files Involved

Core KG extraction:

- `src/kg_node_list.py`
- `prompts/kg_node_list.txt`
- `prompts/kg_relationships.txt`

Neo4j export:

- `src/kg_neo4j.py`

Pipeline wiring:

- `main.py`
- `src/log_utils.py`
- `src/redo_utils.py`
- `src/storage_utils.py`

Dependency:

- `requirements.txt`

## CLI

Run only the KG stage:

```bash
python main.py process --video "Young Sheldon - First Day of High School.mp4" --redo-only kg_extract
```

Run KG and downstream stages:

```bash
python main.py process --video "Young Sheldon - First Day of High School.mp4" --redo kg_extract
```

## Summary

The current Kairos knowledge graph design is based on a few deliberate decisions:

- keep the graph separate from synopsis and RAG
- extract global nodes from short summaries
- extract scene relationships from long descriptions
- use line-based LLM relationship output instead of forcing exact ids
- store minimal relationship JSON in the checkpoint
- use `scene_id` as Neo4j relationship provenance
- create one Neo4j database per video
- rebuild that database from scratch on rerun

This keeps the graph simple enough to debug, structured enough to export, and isolated enough not to interfere with the rest of the Kairos pipeline.
