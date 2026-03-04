# 🌍 Multilingual Transcription Strategy
### Native Transcription vs. Forced English Translation in Kairos RAG

In the Kairos architecture, the pipeline merges Visuals (BLIP/YOLO), Sounds (AST), and Speech (ASR) into an LLM to generate a single *Scene Description*. This description is embedded into a Vector Database for Retrieval-Augmented Generation (RAG).

When processing videos containing non-English speech (e.g., Arabic), we face a structural choice for the ASR stage (Whisper):
1. **Multilingual (Native)**: Transcribe the language exactly as spoken.
2. **Forced English Translation**: Force Whisper to translate all detected foreign speech into English.

This document analyzes the Trade-offs of both approaches within the context of the Kairos RAG system.

---

## Option A: Multilingual (Native Transcription)
*Whisper transcribes the audio in whatever language is spoken natively (e.g., Arabic audio becomes Arabic text).*

### 👍 Pros
1. **Zero Data/Quote Loss**: You capture the *exact* words spoken. If a user queries the RAG system for a specific, verbatim Arabic quote (`"مرحباً"`), the system guarantees a 100% lexical match.
2. **Higher ASR Accuracy**: Whisper is significantly more accurate when transcribing natively compared to translating on the fly. Translation requires Whisper to perform two complex tasks simultaneously, which increases the likelihood of skipped sentences or summarizing instead of verbatim transcription.
3. **Seamless LLM Synergy**: Modern LLMs (like GPT-4o) are inherently bilingual. They can seamlessly read an English BLIP visual description alongside an Arabic Whisper transcript to synthesize a highly detailed, cohesive scene description without requiring external translation steps.

### 👎 Cons
1. **Cross-Lingual Search Dependence**: If the final stored Scene Description contains mixed languages, standard keyword searches (BM25) will struggle. An English query ("finding the water scene") requires the Vector Embedding model to semantically map the English word "water" to the Arabic word in the text. This necessitates a high-quality, multilingual embedding model (e.g., OpenAI's `text-embedding-3-large`).

### 💻 Code Changes Required
No changes needed for Whisper (this is our current default). However, the downstream LLM processing step must be updated to expect potentially non-English text.

**LLM Prompt Update (Example):**
```python
prompt = f"""
You are an AI generating a scene description for a video.
Visual Context: {blip_text}
Audio Context (may be non-English): {whisper_text}

Task: Write a cohesive Scene Description in English. 
If the Audio Context contains foreign speech, include the English translation in the description, but also save the exact original foreign quote in an 'original_quote' field.
"""
```

### ⚖️ Overhead Analysis
- **Compute Overhead**: **Low**. Whisper runs in transcription mode, which is its fastest and most reliable state.
- **LLM Overhead**: **Moderate**. The downstream LLM must be powerful enough (e.g., GPT-4o) to natively understand the dual-language input and format the quote accurately.
- **Infrastructure Overhead**: **Moderate to High**. Requires a multilingual vector embedding model, which is generally more expensive to query than English-only embedding models.


---

## Option B: Forced English Translation
*Whisper is restricted to `language="en"`, forcing it to automatically translate foreign speech into English on the fly.*

### 👍 Pros
1. **Pipeline Uniformity**: All data sent to the downstream LLM is monolithic. BLIP is English, YOLO is English, AST is English, and Whisper output becomes strictly English.
2. **Simpler/Cheaper Database Retrieval**: Because the vector database is strictly English, you can utilize smaller, cheaper, and faster English-only embedding models for RAG search.
3. **Flawless Hybrid Search**: Traditional keyword searches (Elasticsearch/BM25) work flawlessly alongside Vector search because there are no foreign characters or right-to-left layout quirks to break tokenization.

### 👎 Cons
1. **Lost Verbatim Quotes**: A user can *never* search the database for an exact Arabic phrase they heard in the video. The original phrase is permanently discarded in favor of a rough English equivalent.
2. **The "Whisper Translation Tax"**: When Whisper transcribes and translates simultaneously, its hallucination rate spikes dramatically in noisy environments, leading to fabricated sentences or massive chunks of dropped dialogue.

### 💻 Code Changes Required
To force translation, we must bypass the multilingual detector and either lock the Whisper API `language` parameter to `"en"`, or switch endpoints. If using the Azure/OpenAI translation endpoint specifically, the API call changes from `transcriptions` to `translations`.

**Whisper API Update (`whisper_singlecall.py`):**
```python
# To force English translation on all generated chunks:
response = _azure_client.audio.translations.create( 
    model=AZURE_DEPLOYMENT,
    file=audio_file,
    response_format="verbose_json"
)
```

### ⚖️ Overhead Analysis
- **Compute Overhead**: **Moderate to High**. Whisper's internal translation process requires more complex decoding iterations, increasing token generation latency.
- **LLM Overhead**: **Low**. The scene-generation LLM only has to parse English text, allowing you to use smaller, faster, "cheap" LLMs (like GPT-4o-mini).
- **Infrastructure Overhead**: **Low**. The vector database and embedding models can be purely English-based (e.g., `text-embedding-3-small`), significantly reducing long-term vector storage and search costs.


---

## 🏆 Final Recommendation for Kairos

**We strongly recommend Option A (Multilingual/Native Transcription).** 

Because the Kairos architecture uses an LLM at the very end of the pipeline to generate the final *Scene Description*, we should let that LLM do the heavy lifting:

1. Allow Whisper to transcribe the native Arabic flawlessly.
2. Feed the LLM the native Arabic Whisper text along with the English BLIP/YOLO text.
3. Prompt the LLM to write the final Scene Description in English, but instruct it to store the original Arabic dialogue in a sub-field (e.g., `"original_quote": "مرحباً"`).

**Result**: You get the best of both worlds—a unified English database for easy, cheap vector similarity search, while perfectly preserving the exact foreign language quotes for direct keyword lookups and historical accuracy.
