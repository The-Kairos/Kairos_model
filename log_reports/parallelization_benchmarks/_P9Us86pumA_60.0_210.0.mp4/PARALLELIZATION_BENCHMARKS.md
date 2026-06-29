# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 23:28:12 UTC | _P9Us86pumA_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 210.368 | 0.840 | 75.481 | 14.636 | 11.835 | 16.961 | 6.018 |

## 2026-06-25 23:28:12 UTC | _P9Us86pumA_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/_P9Us86pumA_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `210.368` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.840 |
| save_clips | - |
| sample_frames | 1.481 |
| caption_frames | 66.214 |
| sample_fps | 2.622 |
| detect_object_yolo | 12.768 |
| audio_scan | 15.088 |
| asr_timings | 11.235 |
| ast_timings | 49.150 |
| describe_scenes | 14.636 |
| summarize_scenes | 11.835 |
| synthesize_synopsis | 16.961 |
| make_embedding | 6.018 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 67.701 |
| branch_yolo_total | 15.396 |
| branch_audio_total | 75.481 |
