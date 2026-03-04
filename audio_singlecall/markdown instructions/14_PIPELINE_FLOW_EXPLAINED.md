# 🌊 The Kairos Audio Pipeline Flow Explained
### How We Process Audio Efficiently and Accurately (In Plain English)

This document explains the exact step-by-step journey of a video's audio track as it moves through the Kairos pipeline. It details how we achieve massive speedups, eliminate hallucinations, and guarantee that no genuine transcription is ever lost.

---

## 🎬 Step 1: The Visual Foundation (PySceneDetect)
Before we ever touch the audio, the pipeline relies on the visual processing step. 
1. **Scene Chopping:** The video is visually chopped into individual scenes (e.g., Scene 1: 00:00 - 00:05, Scene 2: 00:05 - 00:12).
2. **The Backbone:** These exact timestamps form the "backbone" of our entire audio process. Our ultimate goal is to figure out exactly what was said and heard during each specific visual scene.

---

## 🔎 Step 2: The Pre-Scan & Voice Detection (`audio_detector.py`)
Instead of blindly sending hours of audio to the AI, we aggressively scan it first locally.
1. **Audio Extraction:** We strip the entire audio track from the video in one continuous piece.
2. **Silero VAD (Voice Activity Detection):** We mathematically scan the waveform to find exactly where human voices are. If a 10-minute video only has 2 minutes of talking, we instantly map out those 2 minutes.
3. **Global Language Check:** We randomly sample 5 spots where people are talking. 
   - If we hear a foreign language (like Arabic) at least *twice*, we flag the video as **Multilingual**.
   - If we only hear English, we **Hard-Lock** the video to English so the AI can never hallucinate a foreign language later.
4. **Speech Masking:** We create a secondary version of the audio where all human voices are muted. This is saved for the AST (background noise) model later, ensuring it doesn't get confused by people talking.

---

## 🗣️ Step 3: Transcription (`whisper_singlecall.py`)
This is where the magic happens. We **do not** send the audio scene-by-scene (which is incredibly slow). Instead, we use highly efficient Cloud Chunking.

### Part A: The Parallel API Chunks
1. **The 10-Minute Slices:** We take the entire continuous audio track and slice it into 10-minute (600-second) chunks, with a slight overlap so no words are cut in half.
2. **Parallel Dispatch:** Instead of doing Chunk 1, then Chunk 2, etc., we send up to 4 chunks to the Azure Cloud API **at the exact same time**.
3. **The API Call:** We ask Azure Whisper to transcribe the chunks. 
   - If the video is **Locked** to English, we explicitly tell Azure: "Only output English."
   - If it's **Multilingual**, we tell Azure: "Transcribe exactly what you hear in the native language."
4. **Rate Limit Safety:** If we send chunks too fast and Azure says "Wait!" (Error 429), our workers politely pause for 65 seconds and try again. No data is lost.

### Part B: The Hallucination Gauntlet
When Azure hands the text back, we aggressively filter out the garbage before keeping it:
1. **Confidence Floor:** If Azure isn't highly confident in what it heard (`logprob < -0.9` or `no_speech > 0.6`), we throw it out. It was likely just guessing at background noise.
2. **Emoji Stripper:** We mathematically delete any emojis (🌸), musical notes (♪♫), or bizarre unicode symbols (♥️).
3. **The Loop Cleanser:** If Azure got stuck stuttering (`"Bye. Bye. Bye. Bye."`), we intelligently collapse the sentence down to just `"Bye."` without deleting the rest of the actual dialogue.

### Part C: The Mathematical Re-Mapping (The Most Efficient Step)
1. **The Giant Transcript:** We stitch all the filtered chunks back together into one massive, time-stamped master transcript for the entire video.
2. **Mapping Back to Scenes:** We look at the visual backbone from Step 1. We mathematically ask: *"For Scene 5 (00:30 to 00:45), what words in the Master Transcript fall between these two seconds?"*
3. **The Result:** The words are instantly snapped into their corresponding scenes. If a scene had no talking (or the text was filtered out as a hallucination), the scene's `audio_speech` box is simply safely left blank. No scenes are deleted!

---

## 🚗 Step 4: Background Noise Classification (`ast_parallel.py`)
While Whisper handles the human voices, we also need to know what the environment sounds like (e.g., sirens, birds, car engines).

1. **Parallel Scenes:** We use multiple CPU workers to analyze the scenes simultaneously.
2. **The Speech-Masked Audio:** We specifically feed AST the "muted voices" audio file we made in Step 2. This ensures the model purely hears the car engine, not the people talking over it.
3. **Confidence Filtering:** If the model guesses "Wind" but isn't very sure, we throw the guess away. It must pass a strict confidence threshold.
4. **The Result:** The background sounds are injected directly into the `audio_natural` box for each matching scene.

---

## ✅ Step 5: The Final Output
By the end of this flow, every single visual scene from PySceneDetect is perfectly paired with:
*   `audio_speech`: The pure, hallucination-free transcription (in its native language).
*   `audio_natural`: The confident background noise classification.

This is all packaged into the final `audio_results.json`, ready to be handed to the LLM to generate the final Vector Database Scene Description!
