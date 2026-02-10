# Kairos

Kairos is a video understanding platform designed to analyze **long-form videos** by combining visual and audio context. Its goal is to provide detailed scene-level understanding and enable **clip retrieval based on user queries**.

---

## Features

* **Scene Segmentation:**

  * Uses PySceneDetect to split videos into meaningful scenes.
* **Visual Analysis per Scene:**

  * Frames are sampled from each scene.
  * Each frame is captioned using a lightweight Visual Language Model (BLIP).
  * Object detection is applied to each frame for fine-grained understanding.
* **Audio Analysis per Scene:**

  * AST (Audio Spectrogram Transformer) generates descriptions of natural sounds.
  * Whisper ASR transcribes speech in the scene.
* **Scene Description Integration:**

  * Combines visual captions, object detections, sound descriptions, and speech transcriptions.
  * Uses an LLM to generate a comprehensive description of each scene.
* **Output:**

  * Saves structured data per scene in JSON format.
  * Logs each processing step for reproducibility and debugging.

---

## Usage

```bash
python _download_videos.py          # download videos
python main.py                      # process the videos
python log_reports\_print_logs.py   # see hardware metrics of video processing
```
