# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 23:49:01 UTC | Ejf572xg02M_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 83.025 | 0.845 | 34.509 | 3.856 | 11.312 | 11.376 | 1.563 |

## 2026-06-24 23:49:01 UTC | Ejf572xg02M_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Ejf572xg02M_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `83.025` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.845 |
| save_clips | - |
| sample_frames | 0.173 |
| caption_frames | 9.963 |
| sample_fps | 1.780 |
| detect_object_yolo | 6.272 |
| audio_scan | 16.009 |
| asr_timings | 11.148 |
| ast_timings | 7.343 |
| describe_scenes | 3.856 |
| summarize_scenes | 11.312 |
| synthesize_synopsis | 11.376 |
| make_embedding | 1.563 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 10.141 |
| branch_yolo_total | 8.059 |
| branch_audio_total | 34.509 |
