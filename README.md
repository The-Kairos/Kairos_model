# ✅ **HOW OUR AUDIO PIPELINE WORKS **

_(With Whisper, AST, and Silero VAD explanations)_

Our project extracts two kinds of audio information:

1. **Speech transcription** → using Whisper
2. **Environmental / natural sound detection** → using AST (Audio Spectrogram Transformer)
3. **Speech boundary detection** → using Silero VAD

Everything runs **locally** inside our Python code — no API calls.

---

# 🎬 **1. Scene → Audio Extraction (FFmpeg)**

For every video scene, we cut only the audio that belongs to that scene:

```python
extract_scene_audio_ffmpeg(video_path, scene_03.wav, start, end)
```

This creates files like:

```
output/audio/scene_03.wav
```

These become the inputs for ASR + AST.

---

# 🔊 **2. Speech Recognition (ASR) — HOW WHISPER ACTUALLY WORKS**

### ✔ We install Whisper locally using:

```bash
pip install openai-whisper
```

### ✔ Whisper GitHub

[https://github.com/openai/whisper](https://github.com/openai/whisper)

### ✔ What Whisper is

Whisper is a deep-learning speech recognition model trained on **680,000 hours** of multilingual audio.
It runs locally on the GPU/CPU. No internet is needed.

### ✔ How we use Whisper

1. We load a Whisper model locally (e.g., `medium`):

```python
model = whisper.load_model("medium")
```

2. We only feed Whisper the **speech-only** audio extracted by VAD.

3. Whisper returns a text transcription.

4. We apply a small filter to remove hallucinated endings like:

   - “Thank you for watching”
   - “Thanks”
   - “Thank you”

---

# 🔍 **3. Where Whisper Fits: ASR Pipeline Steps**

### **Step 1 — Load audio**

Load as 16kHz mono.

### **Step 2 — Noise Reduction**

Using `noisereduce` to remove hiss/hum → helps reduce hallucination.

### **Step 3 — Detect speech using VAD (Silero)**

We use **Silero VAD**, a lightweight neural model from GitHub:

### ✔ Silero VAD GitHub

[https://github.com/snakers4/silero-models](https://github.com/snakers4/silero-models)

Silero VAD tells us **where someone is actually speaking**:

```
[
  {"start": 1200, "end": 2400},
  {"start": 5000, "end": 6800}
]
```

These timestamps are in **samples**, not seconds.

### **Step 4 — Extract those speech chunks**

We concatenate the speech into one waveform.

### **Step 5 — Whisper transcribes**

Whisper produces text.

### **Step 6 — Return output**

We save:

```
output/captions/scene_03_asr.txt
```

---

# 🌳 **4. Natural Sound Detection (AST) — HOW AST ACTUALLY WORKS**

We use a HuggingFace model:

### ✔ AST HuggingFace repo

[https://huggingface.co/microsoft/ast](https://huggingface.co/microsoft/ast)

AST = **Audio Spectrogram Transformer**
It is trained on **AudioSet** (2 million sound clips, 527 classes).

### ✔ How AST works

1. Convert audio → log-mel spectrogram
2. Feed spectrogram to transformer
3. Model predicts probabilities for each sound label
4. We keep labels above threshold (e.g., 0.30)

### ✔ What AST detects

527 environmental audio classes, including:

- music
- applause
- crowd noise
- ping
- footsteps
- wind
- traffic
- laughter

AST does **not** detect human speech content → that’s Whisper’s job.

---

# 🔊 **5. Where AST Fits: AST Pipeline Steps**

### **Step 1 — Load audio**

Same as ASR.

### **Step 2 — Mask out speech using Silero VAD**

We remove human speech from the audio:

```python
y_masked[start:end] = 0.0
```

This ensures AST focuses on **environmental sounds** only.

### **Step 3 — Split audio into 2-second clips**

```
[0–2s], [2–4s], [4–6s], ...
```

Each clip is analyzed separately.

### **Step 4 — Extract AST features**

Transform audio → spectrogram → embeddings.

### **Step 5 — AST classifies each clip**

We collect:

- detected labels (e.g., "Music", "Applause")
- confidence scores

### **Step 6 — Save results**

```
output/audio_labels/scene_03_audio_labels.json
```

---

# 📄 **6. Example AST Output File (From Our System)**

Your example explained:

```json
[
  {
    "clip_index": 0,
    "start_sec": 0.0,
    "end_sec": 2.0,
    "labels": [],
    "scores": []
  },
  {
    "clip_index": 1,
    "start_sec": 2.0,
    "end_sec": 4.0,
    "labels": ["Music"],
    "scores": [0.5993]
  },
  {
    "clip_index": 2,
    "start_sec": 4.0,
    "end_sec": 6.0,
    "labels": ["Ping"],
    "scores": [0.3767]
  },
  {
    "clip_index": 6,
    "start_sec": 12.0,
    "end_sec": 14.0,
    "labels": ["Applause"],
    "scores": [0.6789]
  },
  {
    "clip_index": 7,
    "start_sec": 14.0,
    "end_sec": 15.8,
    "labels": ["Applause"],
    "scores": [0.7377]
  }
]
```

This means:

- Music happens around 2–4 seconds
- A ping sound at 4–6 seconds
- Applause around 12–16 seconds
- Other segments contain no meaningful environmental sounds

---

# 🧩 **7. Final Combined Caption (BLIP + ASR + AST)**

Your final caption is constructed by concatenating:

### **BLIP (visual)**

“A video frame of a woman speaking at a podium…”

### **ASR (speech)**

"I'm proud… to receive this award."

### **AST (environmental audio)**

Music, Ping, Applause, Applause

### ✔ Final combined caption:

```
BLIP: a video frame of a woman speaking at a podium +
ASR: I'm proud, well in fact I'm very proud, to be the first Pashtun, the first Pakistani, and the youngest person to receive this award. +
AST: Music, Ping, Applause, Applause
```

Every modality contributes:

| Component | Purpose              |
| --------- | -------------------- |
| **BLIP**  | What the camera sees |
| **ASR**   | What humans say      |
| **AST**   | Background sounds    |

Together, they form the **full scene understanding**.

---

# 🚀 **8. Why We Add Cloud / API-Based Audio Processing (NEW)**

While our **core pipeline runs fully locally** using Whisper, AST, and Silero VAD, we additionally introduce **API-based audio processing modules** for **deployment and scalability**.

This section explains **why** these new components are added and **what problem they solve**.

---

## 🎯 **Motivation for Adding API-Based Audio Processing**

Our local pipeline is ideal for:

- Research experiments
- Model inspection and debugging
- Offline processing
- Academic reproducibility

However, **real-world deployment** introduces constraints:

| Challenge        | Local Models      | Cloud APIs       |
| ---------------- | ----------------- | ---------------- |
| GPU availability | Required          | Not required     |
| Deployment       | Heavy Docker / VM | Lightweight      |
| Scaling          | Limited           | Automatic        |
| Maintenance      | Manual updates    | Provider-managed |
| Cost control     | Hardware-bound    | Usage-based      |

To address this, we **add cloud-compatible alternatives** while **preserving the same logical pipeline**.

---

## 🔄 **What Changes vs What Stays the Same**

### ✅ What stays the same (Conceptually)

- Scene-level audio extraction
- Separation of **speech** and **non-speech**
- Scene-based captions
- Final multi-modal fusion (BLIP + ASR + AST-style labels)

---

### 🔁 What changes (Implementation only)

| Task                 | Local Version   | Cloud Version                 |
| -------------------- | --------------- | ----------------------------- |
| Speech Recognition   | Whisper (local) | Whisper API                   |
| VAD                  | Silero VAD      | Whisper timestamps            |
| Environmental Sounds | AST (local)     | AudioSet-compatible Audio API |
| Execution            | PyTorch         | HTTPS API calls               |

---

## 🔊 **9. Cloud-Based Speech Recognition (Whisper API)**

### ✔ Why we add it

- Removes need for local GPU
- Uses the **same Whisper model family**
- Provides built-in speech segmentation
- More stable for long-form audio

### ✔ What it replaces

| Local                  | Cloud              |
| ---------------------- | ------------------ |
| Silero VAD             | Whisper timestamps |
| `whisper.load_model()` | Whisper API        |
| PyTorch inference      | API request        |

### ✔ Result

- Same output: **transcribed speech**
- Same role in pipeline: **ASR**
- Different execution: **remote inference**

---

## 🌍 **10. Cloud-Based Environmental Audio Detection (AST Replacement)**

### ⚠ Important clarification

**AST does not have an official API.**

Therefore, in deployment we use **AudioSet-compatible cloud services** that produce:

- The **same 527 AudioSet labels**
- Probabilities per sound class
- Segment-level predictions

### ✔ Why this is valid

- AST itself is trained on **AudioSet**
- AudioSet defines the **label ontology**, not AST
- Cloud services use the **same label space**

Thus, the output semantics remain identical.

---

## 🌳 **11. Why We Still Call It “AST-Style Audio Detection”**

Even in cloud mode:

- Labels come from **AudioSet**
- Detection focuses on **non-speech environmental sounds**
- Speech is excluded before classification
- Output format remains unchanged

So the **logical role of AST** is preserved, even though the backend changes.

---

## 🧠 **12. Dual-Mode Design: Research vs Deployment**

Our system supports **two execution modes**:

| Mode           | Purpose                             |
| -------------- | ----------------------------------- |
| **Local Mode** | Research, debugging, experiments    |
| **Cloud Mode** | Deployment, scalability, production |

Example switch:

```python
USE_CLOUD_AUDIO = True
```

This allows us to:

- Benchmark local AST vs cloud labels
- Validate consistency
- Move from research → production cleanly

---

## 📁 **13. Updated Audio Module Structure**

```
src/
├── audio_asr.py              # Local Whisper + Silero VAD
├── audio_natural.py          # Local AST
│
├── audio_asr_api.py          # Whisper API (cloud)
├── audio_natural_api.py      # AudioSet-compatible API
│
├── audio_utils.py
├── frame_captioning_blip.py
├── scene_cutting.py
└── main.py
```

---

## 🧩 **14. Final Pipeline (Unified View)**

Whether running locally or via APIs, the **logical pipeline remains identical**:

```
Scene Audio
   ├── ASR → What is said
   ├── Environmental Audio → What is heard
   └── Vision → What is seen
```

The difference lies **only in execution**, not in **understanding**.

---

## ✅ **Key Takeaway**

> We add cloud-based audio processing **not to replace our models**,
> but to **extend our system from research to real-world deployment**
> while keeping the **same semantic pipeline and outputs**.

---
