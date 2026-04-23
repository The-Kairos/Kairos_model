# Kairos Refactor Guide

This file defines how agents should refactor Kairos. Keep the code simple,
functional, model-agnostic, and easy to read.

## Core Rules

- Use KISS: choose the simplest readable design that works.
- One function should have one responsibility.
- Keep functions under five lines whenever practical.
- Prefer flat logic over nested `if` statements.
- Name things so clearly that the logic becomes obvious.
- Separate the "what" from the "how".
- Extract repeated logic into helper functions.
- Use data structures instead of long branching chains.
- Avoid hidden side effects.
- Avoid giant functions.
- Avoid unclear helper names.
- Avoid mixing unrelated responsibilities.
- Avoid premature generalization.
- Prefer clarity over DRY when DRY makes code harder to read.
- Do not use Python OOP for application logic.
- Do not add classes.
- Do not add test classes, typing-only classes, or protocol/helper classes when a function or plain dictionary is enough.
- Do not combine unrelated domains just to reduce duplication.

Example: audio and video should stay separate. Do not merge audio and video
logic into one abstraction only because both have a `process_*` function.

## Functional Style

Write code as small functions that accept data and return data.

Good:

```python
def create_chunk(start, end):
    return {
        "start": start,
        "end": end,
    }
```

Avoid functions that read global state, mutate unrelated objects, perform IO,
and transform data all at once.

Good shape:

```python
def build_summary_prompt(text):
    return f"Summarize:\n{text}"
```

```python
def summarize(text, llm_fn):
    prompt = build_summary_prompt(text)
    return llm_fn(prompt)
```

## No Python OOP

Do not introduce classes, inheritance trees, service objects, managers, or
framework-style objects for Kairos logic.

Use functions, dictionaries, tuples, dataclasses only if needed for plain data,
and small modules grouped by responsibility.

This no-class rule also applies to tests and lightweight typing helpers.
Prefer plain test functions, plain dictionaries, and small schema builders.

Prefer this:

```python
def transcribe_audio(audio_path, asr_fn):
    return asr_fn(audio_path)
```

Avoid this:

```python
class AudioTranscriber:
    def __init__(self, model):
        self.model = model

    def transcribe(self, audio_path):
        return self.model(audio_path)
```

## SOLID With Functional Code

Apply SOLID principles without classes.

### Single Responsibility Principle

Each function does one task.

Good:

```python
def extract_audio(video_path):
    ...
```

```python
def transcribe_audio(audio_path, asr_fn):
    return asr_fn(audio_path)
```

Avoid one function that extracts audio, transcribes it, summarizes it, writes
logs, and updates a database.

### Open Closed Principle

Add new behavior by passing new functions or extending registries, not by
editing every caller.

Good:

```python
def summarize(text, llm_fn):
    prompt = build_summary_prompt(text)
    return llm_fn(prompt)
```

Adding Gemini should mean adding `gemini_llm`, not rewriting `summarize`.

### Liskov Substitution Principle

Any injected function with the same contract should be swappable.

Good:

```python
def gpt_llm(prompt):
    ...

def gemini_llm(prompt):
    ...
```

Both should accept the same input shape and return the same output shape.
Callers should not care which provider is used.

### Interface Segregation Principle

Functions should depend only on the small callable they need.

Good:

```python
def caption_video_frames(frame_paths, vlm_fn):
    return [vlm_fn(frame_path) for frame_path in frame_paths]
```

Avoid passing a large provider object or config blob when the function only
needs one callable.

### Dependency Inversion Principle

High-level pipeline logic depends on injected functions, not provider details.

Good:

```python
def process_video(video_path, vlm_fn, asr_fn, llm_fn):
    ...
```

Avoid importing OpenAI, Gemini, Whisper, YOLO, or BLIP inside orchestration
logic unless that file is specifically a model adapter.

## Model-Agnostic Pipeline Rules

The pipeline must be model-agnostic.

- Models are injectable.
- Pipeline logic must not know provider details.
- Provider-specific code belongs in model adapter modules.
- Prompts must not be mixed with API call logic.
- Reuse the same prompts across LLM providers.
- Model calls must return consistent formats.
- Do not leak provider details into pipeline modules.

Clean mental model:

```python
def build_summary_prompt(text):
    return f"Summarize:\n{text}"
```

```python
def summarize(text, llm_fn):
    prompt = build_summary_prompt(text)
    return llm_fn(prompt)
```

```python
def gpt_llm(prompt):
    ...
```

```python
def gemini_llm(prompt):
    ...
```

Models are injectable:

```python
process_video(
    video_path,
    vlm_fn=blip_caption,
    asr_fn=whisper_transcribe,
    llm_fn=gpt_describe,
)
```

A future pipeline orchestrator should not do model work itself. It should only
connect steps. Do not build the full orchestrator yet; first refine the modules.

## LLM Registry Pattern

Use a registry to select models without leaking provider logic.

```python
def get_llms():
    return {
        "gpt": gpt_llm,
        "gemini": gemini_llm,
        "local": local_llm,
    }
```

```python
def get_llm(name):
    return get_llms()[name]
```

Environment-based default:

```python
import os


def get_default_llm():
    if os.getenv("OPENAI_API_KEY"):
        return gpt_llm

    return local_llm
```

If users provide API keys, use API models. If they do not, try to run locally.

## Safe Model Calls

Wrap model calls safely and consistently.

```python
import time


def llm_call(llm_fn, prompt, retries=3):
    for attempt in range(retries):
        try:
            return llm_fn(prompt)
        except Exception as error:
            if attempt == retries - 1:
                raise error
            time.sleep(1)
```

Keep retry logic outside provider adapters unless the provider SDK requires a
specific local behavior.

## Timing And Logging

Use timing wrappers in `src/kairos/logging/timing.py`.

Example:

```python
@log_step()
def caption_frames_log(*args, **kwargs):
    return caption_frames(*args, **kwargs)
```

Logging code must not change business logic. It should observe and record.

Reusable output shapes should live in `src/kairos/logging/schemas.py` when more
than one module needs the same dictionary structure. Do not duplicate schema
builders across modules.

## Target File Structure

Refactor toward this structure:

```text
_jsonkairos/
├── main.py
├── pyproject.toml
├── README.md
├── package.json
├── src/
│   └── kairos/
│       ├── pipelines/
│       │   ├── audio.py
│       │   ├── rag.py
│       │   ├── summarization.py
│       │   └── video.py
│       ├── modules/
│       │   ├── frame_captions.py
│       │   ├── natural_sounds.py
│       │   ├── spatio_temporal.py
│       │   └── speech_transcription.py
│       ├── models/
│       │   ├── _registry.py
│       │   ├── asr.py
│       │   ├── ast.py
│       │   ├── gemini.py
│       │   ├── gpt.py
│       │   ├── whisper.py
│       │   └── yolo.py
│       ├── logging/
│       │   ├── checkpoint.py
│       │   ├── io.py
│       │   ├── schemas.py
│       │   └── timing.py
│       ├── utils/
│       └── cli/
│           └── main.py
└── tests/
```

`utils/` is only for helpers reused across multiple areas. Do not put domain
logic there just because no better name exists yet.

## Package Rules

Do not put application logic in `__init__.py`.

Keep `__init__.py` empty unless packaging requires a tiny export. Prefer
explicit module imports and `pyproject.toml` configuration.

Prefer intuitive docstrings on public modules and public functions. Keep them
short and practical:

- explain the function's responsibility
- state the input/output contract when not obvious
- note retry or fallback behavior when that affects downstream logic

In `pyproject.toml`, expose CLI entry points through `[project.scripts]` so
these commands are possible:

```bash
kairos process <video_path>
kairos download-test-video
kairos rag <video_id>
```

Example:

```toml
[project.scripts]
kairos = "kairos.cli.main:main"
```

The CLI should route subcommands like `process`, `download-test-video`, and
`rag`.

## NPM App Script

If `package.json` is present, include:

```json
{
  "scripts": {
    "app": "python -m kairos.cli.main process data/raw/input.mp4"
  }
}
```

Users should be able to run:

```bash
npm run app
```

The app should prefer API models when keys exist and local fallbacks when keys
are missing.

## Prompt Rules

- Prompts belong in prompt-building functions or prompt modules.
- API calls belong in model adapter modules.
- Do not hardcode provider names into prompt logic.
- Do not duplicate prompts across providers.

Good:

```python
def build_video_summary_prompt(captions, transcript):
    return f"Captions:\n{captions}\n\nTranscript:\n{transcript}"
```

```python
def describe_video(captions, transcript, llm_fn):
    prompt = build_video_summary_prompt(captions, transcript)
    return llm_call(llm_fn, prompt)
```

## Branching Rules

Prefer data structures over branching.

Good:

```python
COMMANDS = {
    "process": process_command,
    "rag": rag_command,
    "download-test-video": download_test_video_command,
}
```

```python
def run_command(name, args):
    return COMMANDS[name](args)
```

Avoid:

```python
if name == "process":
    ...
elif name == "rag":
    ...
elif name == "download-test-video":
    ...
```

Use guard clauses to keep logic flat.

Good:

```python
def require_video_path(video_path):
    if not video_path:
        raise ValueError("video_path is required")
```

## Testing Rules

Use three levels of tests:

- Unit tests: mock model calls.
- Smoke tests: run real models on tiny sample inputs.
- Integration tests: verify pipeline steps connect correctly.

Organize tests by folder:

- `test/unit`
- `test/smoke`
- `test/integration`

Prefer function-based tests with plain `assert` over `unittest.TestCase`
classes unless a library requires a different style.

At the top of each runnable test script, add a first-line comment showing how
to run it, for example:

```python
# Run: python test/unit/test_example.py
```

Unit test example:

```python
def fake_llm(prompt):
    return {"text": "short summary"}


def test_summarize_uses_injected_llm():
    result = summarize("hello", fake_llm)
    assert result["text"] == "short summary"
```

Smoke tests should be small, optional when expensive, and safe to skip when
local models or API keys are unavailable.

Integration tests should prove modules connect without embedding provider
details in orchestration code.

## Refactor Order

When refactoring existing code:

1. Identify one responsibility.
2. Extract the smallest useful function.
3. Name the function after what it does.
4. Move provider-specific logic into model adapters.
5. Move prompt text into prompt builders.
6. Add or update focused tests.
7. Stop before introducing broad abstractions.

Do not refactor the whole system at once. Make small, reviewable changes.
