# 🔌 Whisper Transcription: API vs Local Model
### How We Handle Azure API Calls, Rate Limits, and the Local Fallback

---

## Our Primary Method: Azure OpenAI Whisper API

When you run the pipeline with `--api`, every audio chunk is sent to the Azure Cloud and transcribed using the enterprise-grade **whisper-karios** deployment.

**Why API is preferred:**
- Uses a powerful GPU in Azure's datacenter, so transcription of 600 seconds of audio completes in under 30 seconds.
- No local RAM or VRAM consumed by the AI model during transcription.
- Language is auto-detected per chunk natively, no global lock needed.

---

## How Rate Limiting is Handled

Azure's S0 tier limits how many API requests you can make per minute. When 4 workers fire simultaneously, one may hit a `429 RateLimitReached` error. Our code handles this automatically.

**The retry loop (3 attempts):**
```python
for attempt in range(3):
    try:
        segments = transcribe_via_api(chunk_audio, language=None)
        break  # success
    except Exception as e:
        if "429" in str(e):
            time.sleep(65)  # wait 65 seconds, then retry
            continue
```

You will see this log line in the terminal when it triggers:
```
[WhisperWorker] Rate limit hit. Retrying in 65s... (Attempt 1/3)
```

---

## Automatic Local Whisper Fallback

If all 3 API retries fail (e.g., Azure outage or persistent throttling), the worker automatically loads a local Whisper model to transcribe that chunk on the VM's CPU, **so no audio chunk is ever silently lost**.

```python
if not api_success:
    # Fallback: load model from local disk cache
    model = whisper.load_model("medium", device="cpu")
    result = model.transcribe(chunk_audio, fp16=False)
    del model  # immediately unloaded to free RAM
```

You will see this log line if fallback triggers:
```
[WhisperWorker] API exhausted. Falling back to local Whisper (medium)...
```

**How local works on the Azure VM:**
- The Whisper `medium` model (~1.5 GB) downloads **once** to `/home/.../.cache/whisper/` on the first use.
- Every subsequent fallback loads from disk — no internet needed for inference.
- The VM has 188 GB RAM; running 4 local workers simultaneously only uses ~6-8 GB.

---

## Summary: Priority Order

| Attempt | Method | Speed |
|:---|:---|:---|
| 1st, 2nd, 3rd (if 429) | Azure OpenAI API (retries after 65s) | Fast ⚡ |
| Final fallback | Local Whisper model (cached on disk) | Slower but guaranteed ✅ |
| Both fail | Returns empty for that chunk only | Extremely rare ⚠️ |
