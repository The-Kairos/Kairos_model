# 🚫 Hallucination Filtering
### How We Clean Whisper Output: Emojis, Noise, Loops, and Low-Confidence Text

The Azure Whisper API is powerful, but it sometimes generates "hallucinated" text — fabricated content that sounds plausible but isn't real speech. This happens most often over background music, crowd noise, or silence. This document explains every layer of filtering we apply.

---

## Layer 1: Emoji & Symbol Stripping

Whisper often injects emojis and musical symbols when it hears background music. We remove all of them using two methods:

**1. The `emoji` library** — strips all unicode emoji characters:
```python
import emoji
text = emoji.replace_emoji(text, replace="")
```

**2. Regex fallback** — catches specific musical/decorative symbols:
```python
import re
text = re.sub(r'[♪♫♥️✿❀❁]+', '', text).strip()
```

**Result:** A segment that was `"♪♫ ♪♫"` becomes completely empty and is **deleted**. A segment that was `"Hello. ♪"` becomes `"Hello."` — the real speech is preserved.

---

## Layer 2: Confidence Score Floor (Logprob)

Every segment from the Azure API includes an `avg_logprob` confidence score. Closer to `0.0` means the model is very confident. Below `-0.9` means it was just guessing.

```python
if seg.get("avg_logprob", 0) < -0.9:
    continue  # drop this segment
```

**Effect:** Muffled background babble and noise artifacts score very low (e.g., `-1.4`) and are silently dropped. Genuine speech (even in noisy environments) scores closer to `-0.2` to `-0.5`.

---

## Layer 3: No-Speech Probability

The API also returns a `no_speech_prob` value — the model's own estimate that the audio contains no actual human voice.

```python
if seg.get("no_speech_prob", 0) > 0.6:
    continue  # model itself says this isn't speech
```

---

## Layer 4: Repetition Loop Collapse

Whisper sometimes gets stuck in a "stuttering loop" — repeating the same phrase over and over within a single segment. The `clean_repetitive_text()` function grammatically detects and collapses these.

**Example:**
```
Input:  "Bye. Bye. Bye. Bye. Bye. One of the coolest streets coming up right here."
Output: "Bye. One of the coolest streets coming up right here."
```

The genuine sentence at the end is **always preserved**. Only the repeated prefix is collapsed.

---

## Layer 5: Cross-Segment Deduplication

If a phrase appears in `audio_speech` already and then appears again in the next segment (an inter-segment loop), we track it  in a `seen_texts` set and skip adding the duplicate:

```python
if text_lower in seen_texts:
    continue
seen_texts.add(text_lower)
```

---

## Layer 6: Noise Character Detection

Whisper sometimes transcribes musical "drone" tones as bizarre extended Latin characters (`Ə Ɓ Ƥ`). If more than 15% of a segment's characters are these symbols, it's flagged as a noise hallucination and dropped.

```python
NOISE_CHARS = set("ƏƁƟƙƒƠƣƜơ")
if special_count / len(text) > 0.15:
    continue
```
