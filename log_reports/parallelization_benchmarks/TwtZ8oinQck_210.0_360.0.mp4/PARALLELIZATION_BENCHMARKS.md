# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 18:10:31 UTC | TwtZ8oinQck_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 122.072 | 0.899 | 43.210 | 8.481 | 16.375 | 12.487 | 2.511 |

## 2026-06-25 18:10:31 UTC | TwtZ8oinQck_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/TwtZ8oinQck_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `122.072` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.899 |
| save_clips | - |
| sample_frames | 0.575 |
| caption_frames | 26.132 |
| sample_fps | 2.061 |
| detect_object_yolo | 7.876 |
| audio_scan | 14.106 |
| asr_timings | 10.251 |
| ast_timings | 18.843 |
| describe_scenes | 8.481 |
| summarize_scenes | 16.375 |
| synthesize_synopsis | 12.487 |
| make_embedding | 2.511 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 26.713 |
| branch_yolo_total | 9.943 |
| branch_audio_total | 43.210 |
