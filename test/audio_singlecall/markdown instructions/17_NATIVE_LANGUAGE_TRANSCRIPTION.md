# 🌐 Native Language Transcription Plan
### How We Transcribe in the Spoken Language Without Forcing Translation

---

## The Core Principle

Whisper can do two different things with non-English audio:
1. **Transcription** — Write down *exactly* what was said in the original language.
2. **Translation** — Read the audio in Arabic (for example) and write the output in English.

**We always use option 1 (Transcription).** The downstream LLM (GPT-4o) is fully fluent in all major world languages and will handle translation when generating the final scene description. Our job is to pass the rawest, most accurate representation of what was actually said.

---

## How It's Implemented

We pass `language=None` to the Azure Whisper API for every chunk. This tells Whisper:
> "You decide what language this audio is. Transcribe it exactly as it is spoken."

```python
# In whisper_singlecall.py → transcribe_via_api()
response = _azure_client.audio.transcriptions.create(
    model=AZURE_DEPLOYMENT,
    file=audio_file,
    language=None,          # <- auto-detect, no forced translation
    response_format="verbose_json"
)
```

Whisper auto-detects the language per 10-minute chunk independently. This means:
- A video that starts with an English intro and switches to Arabic 40 minutes in will be correctly transcribed in each language at the moment it's spoken.
- No global language lock is applied when using the API.

---

## How Different Languages Are Stored in the JSON

The `audio_speech` field in `audio_results.json` stores the text **exactly as Whisper returns it** — in the native script and language. Examples below:

### English (Latin script, LTR)
```json
{
    "scene_index": 14,
    "audio_speech": "Welcome to the graduation ceremony."
}
```

### Arabic (Arabic script, RTL)
```json
{
    "scene_index": 435,
    "audio_speech": "نبدأ مع كلية الأعمال ونطلب من الدكتور راشد الانضمام."
}
```

### Chinese (Simplified, LTR)
```json
{
    "scene_index": 72,
    "audio_speech": "欢迎来到北京，这里是天安门广场。"
}
```

### Filipino / Tagalog (Latin script, LTR)
```json
{
    "scene_index": 8,
    "audio_speech": "Maligayang pagdating sa ating selebrasyon."
}
```

### Mixed (English + Arabic in same scene)
```json
{
    "scene_index": 210,
    "audio_speech": "Please welcome Dr. Salim. أهلاً وسهلاً."
}
```

All scripts — left-to-right, right-to-left, ideographic — are stored exactly as unicode text characters. The LLM reads and understands all of them without any conversion.

---

## Why We Do NOT Force English Translation

Forcing `language="en"` on an Arabic video causes Whisper to translate instead of transcribe. This has two major problems:

1. **Lower accuracy:** Whisper's translation path uses a separate decoding head and is more error-prone than pure transcription.
2. **Data loss:** The exact Arabic phrase is permanently discarded. If a user later queries the RAG system for a specific quote, there is nothing to match against.

The LLM at the end of the Kairos pipeline (GPT-4o) is the right place to do cross-lingual reasoning — it generates the final English scene description while keeping the native-language quote preserved in the stored data.

---

## Summary Table

| Language Spoken | Stored In JSON | LLM Reads |
|:---|:---|:---|
| English | English | ✅ Yes |
| Arabic | Arabic (RTL) | ✅ Yes |
| Chinese | Chinese | ✅ Yes |
| Filipino/Tagalog | Tagalog | ✅ Yes |
| Mixed (En + Ar) | Both scripts | ✅ Yes |
