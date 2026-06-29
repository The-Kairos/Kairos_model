# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 09:13:33 UTC | ibWW_MYY1C8_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 220.406 | 0.737 | 110.376 | 12.201 | 12.299 | 29.222 | 3.293 |

## 2026-06-26 09:13:33 UTC | ibWW_MYY1C8_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ibWW_MYY1C8_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `220.406` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.737 |
| save_clips | - |
| sample_frames | 1.470 |
| caption_frames | 38.408 |
| sample_fps | 2.252 |
| detect_object_yolo | 8.701 |
| audio_scan | 14.072 |
| asr_timings | 69.103 |
| ast_timings | 27.192 |
| describe_scenes | 12.201 |
| summarize_scenes | 12.299 |
| synthesize_synopsis | 29.222 |
| make_embedding | 3.293 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.884 |
| branch_yolo_total | 10.959 |
| branch_audio_total | 110.376 |
