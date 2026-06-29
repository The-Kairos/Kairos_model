# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 09:21:10 UTC | ibWW_MYY1C8_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 252.184 | 0.702 | 102.732 | 23.029 | 28.768 | 26.728 | 4.496 |

## 2026-06-26 09:21:10 UTC | ibWW_MYY1C8_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ibWW_MYY1C8_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `252.184` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.702 |
| save_clips | - |
| sample_frames | 1.337 |
| caption_frames | 50.461 |
| sample_fps | 2.219 |
| detect_object_yolo | 10.297 |
| audio_scan | 15.079 |
| asr_timings | 49.436 |
| ast_timings | 38.208 |
| describe_scenes | 23.029 |
| summarize_scenes | 28.768 |
| synthesize_synopsis | 26.728 |
| make_embedding | 4.496 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.803 |
| branch_yolo_total | 12.521 |
| branch_audio_total | 102.732 |
