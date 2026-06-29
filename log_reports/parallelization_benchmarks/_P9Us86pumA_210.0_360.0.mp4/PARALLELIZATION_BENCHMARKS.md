# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 23:24:40 UTC | _P9Us86pumA_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 175.343 | 0.681 | 57.216 | 14.560 | 21.963 | 13.645 | 4.244 |

## 2026-06-25 23:24:40 UTC | _P9Us86pumA_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/_P9Us86pumA_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `175.343` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.681 |
| save_clips | - |
| sample_frames | 1.163 |
| caption_frames | 49.182 |
| sample_fps | 2.096 |
| detect_object_yolo | 9.163 |
| audio_scan | 13.180 |
| asr_timings | 8.181 |
| ast_timings | 35.847 |
| describe_scenes | 14.560 |
| summarize_scenes | 21.963 |
| synthesize_synopsis | 13.645 |
| make_embedding | 4.244 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 50.351 |
| branch_yolo_total | 11.265 |
| branch_audio_total | 57.216 |
