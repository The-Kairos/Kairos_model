# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 13:49:41 UTC | 7MG8MlfEL-k_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 186.193 | 0.789 | 70.696 | 16.468 | 13.281 | 33.037 | 3.414 |

## 2026-06-24 13:49:41 UTC | 7MG8MlfEL-k_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/7MG8MlfEL-k_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `186.193` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.789 |
| save_clips | - |
| sample_frames | 1.099 |
| caption_frames | 35.470 |
| sample_fps | 2.270 |
| detect_object_yolo | 8.271 |
| audio_scan | 10.587 |
| asr_timings | 33.534 |
| ast_timings | 26.567 |
| describe_scenes | 16.468 |
| summarize_scenes | 13.281 |
| synthesize_synopsis | 33.037 |
| make_embedding | 3.414 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.575 |
| branch_yolo_total | 10.548 |
| branch_audio_total | 70.696 |
