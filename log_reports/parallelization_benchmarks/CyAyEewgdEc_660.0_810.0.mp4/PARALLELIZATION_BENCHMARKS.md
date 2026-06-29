# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 22:17:43 UTC | CyAyEewgdEc_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 371.795 | 0.776 | 277.683 | 14.393 | 7.720 | 10.086 | 3.794 |

## 2026-06-24 22:17:43 UTC | CyAyEewgdEc_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/CyAyEewgdEc_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `371.795` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.776 |
| save_clips | - |
| sample_frames | 1.187 |
| caption_frames | 43.408 |
| sample_fps | 2.319 |
| detect_object_yolo | 8.993 |
| audio_scan | 11.880 |
| asr_timings | 233.024 |
| ast_timings | 32.770 |
| describe_scenes | 14.393 |
| summarize_scenes | 7.720 |
| synthesize_synopsis | 10.086 |
| make_embedding | 3.794 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.601 |
| branch_yolo_total | 11.318 |
| branch_audio_total | 277.683 |
