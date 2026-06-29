# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 15:36:21 UTC | l_VtY7btNRA_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 193.778 | 0.814 | 53.695 | 22.509 | 37.091 | 25.620 | 3.352 |

## 2026-06-26 15:36:21 UTC | l_VtY7btNRA_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/l_VtY7btNRA_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `193.778` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.814 |
| save_clips | - |
| sample_frames | 0.919 |
| caption_frames | 37.439 |
| sample_fps | 2.249 |
| detect_object_yolo | 8.669 |
| audio_scan | 13.015 |
| asr_timings | 13.424 |
| ast_timings | 27.247 |
| describe_scenes | 22.509 |
| summarize_scenes | 37.091 |
| synthesize_synopsis | 25.620 |
| make_embedding | 3.352 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.364 |
| branch_yolo_total | 10.925 |
| branch_audio_total | 53.695 |
