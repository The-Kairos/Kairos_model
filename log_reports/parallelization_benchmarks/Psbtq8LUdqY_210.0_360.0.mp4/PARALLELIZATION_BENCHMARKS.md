# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 14:33:38 UTC | Psbtq8LUdqY_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 227.984 | 0.778 | 66.792 | 22.484 | 34.344 | 26.560 | 5.136 |

## 2026-06-25 14:33:38 UTC | Psbtq8LUdqY_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Psbtq8LUdqY_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `227.984` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.778 |
| save_clips | - |
| sample_frames | 1.371 |
| caption_frames | 55.674 |
| sample_fps | 2.474 |
| detect_object_yolo | 10.943 |
| audio_scan | 16.720 |
| asr_timings | 9.278 |
| ast_timings | 40.786 |
| describe_scenes | 22.484 |
| summarize_scenes | 34.344 |
| synthesize_synopsis | 26.560 |
| make_embedding | 5.136 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 57.051 |
| branch_yolo_total | 13.422 |
| branch_audio_total | 66.792 |
