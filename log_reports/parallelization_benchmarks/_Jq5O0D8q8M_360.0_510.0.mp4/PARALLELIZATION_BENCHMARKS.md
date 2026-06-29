# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 23:18:44 UTC | _Jq5O0D8q8M_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 165.302 | 0.730 | 59.248 | 14.391 | 14.604 | 6.938 | 4.417 |

## 2026-06-25 23:18:44 UTC | _Jq5O0D8q8M_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/_Jq5O0D8q8M_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `165.302` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.730 |
| save_clips | - |
| sample_frames | 1.601 |
| caption_frames | 49.522 |
| sample_fps | 2.384 |
| detect_object_yolo | 10.036 |
| audio_scan | 9.503 |
| asr_timings | 10.971 |
| ast_timings | 38.765 |
| describe_scenes | 14.391 |
| summarize_scenes | 14.604 |
| synthesize_synopsis | 6.938 |
| make_embedding | 4.417 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.129 |
| branch_yolo_total | 12.425 |
| branch_audio_total | 59.248 |
