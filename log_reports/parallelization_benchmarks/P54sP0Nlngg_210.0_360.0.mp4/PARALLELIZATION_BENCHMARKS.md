# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 12:46:25 UTC | P54sP0Nlngg_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 202.914 | 0.805 | 57.891 | 19.360 | 31.966 | 29.594 | 3.917 |

## 2026-06-25 12:46:25 UTC | P54sP0Nlngg_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/P54sP0Nlngg_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `202.914` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.805 |
| save_clips | - |
| sample_frames | 1.015 |
| caption_frames | 44.909 |
| sample_fps | 2.317 |
| detect_object_yolo | 9.711 |
| audio_scan | 13.297 |
| asr_timings | 11.486 |
| ast_timings | 33.099 |
| describe_scenes | 19.360 |
| summarize_scenes | 31.966 |
| synthesize_synopsis | 29.594 |
| make_embedding | 3.917 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.930 |
| branch_yolo_total | 12.034 |
| branch_audio_total | 57.891 |
