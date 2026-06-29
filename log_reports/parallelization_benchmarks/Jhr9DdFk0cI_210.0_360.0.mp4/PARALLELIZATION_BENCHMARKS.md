# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 05:16:13 UTC | Jhr9DdFk0cI_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 166.575 | 0.893 | 73.349 | 11.751 | 11.511 | 22.104 | 3.016 |

## 2026-06-25 05:16:13 UTC | Jhr9DdFk0cI_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Jhr9DdFk0cI_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `166.575` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.893 |
| save_clips | - |
| sample_frames | 1.348 |
| caption_frames | 30.977 |
| sample_fps | 2.300 |
| detect_object_yolo | 7.872 |
| audio_scan | 16.106 |
| asr_timings | 33.627 |
| ast_timings | 23.608 |
| describe_scenes | 11.751 |
| summarize_scenes | 11.511 |
| synthesize_synopsis | 22.104 |
| make_embedding | 3.016 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.332 |
| branch_yolo_total | 10.177 |
| branch_audio_total | 73.349 |
