# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 13:36:57 UTC | POU_6XcdD1s_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 164.223 | 0.658 | 51.327 | 16.116 | 16.112 | 30.318 | 3.066 |

## 2026-06-25 13:36:57 UTC | POU_6XcdD1s_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/POU_6XcdD1s_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `164.223` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.658 |
| save_clips | - |
| sample_frames | 0.986 |
| caption_frames | 33.655 |
| sample_fps | 2.075 |
| detect_object_yolo | 8.458 |
| audio_scan | 13.354 |
| asr_timings | 13.480 |
| ast_timings | 24.483 |
| describe_scenes | 16.116 |
| summarize_scenes | 16.112 |
| synthesize_synopsis | 30.318 |
| make_embedding | 3.066 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 34.647 |
| branch_yolo_total | 10.539 |
| branch_audio_total | 51.327 |
