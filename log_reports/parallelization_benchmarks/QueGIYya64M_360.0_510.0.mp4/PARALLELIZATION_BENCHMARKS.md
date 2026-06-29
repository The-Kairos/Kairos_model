# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 15:49:59 UTC | QueGIYya64M_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 220.725 | 0.792 | 82.511 | 23.271 | 28.119 | 11.867 | 5.099 |

## 2026-06-25 15:49:59 UTC | QueGIYya64M_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/QueGIYya64M_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `220.725` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.792 |
| save_clips | - |
| sample_frames | 1.494 |
| caption_frames | 52.828 |
| sample_fps | 2.488 |
| detect_object_yolo | 10.750 |
| audio_scan | 13.344 |
| asr_timings | 28.928 |
| ast_timings | 40.230 |
| describe_scenes | 23.271 |
| summarize_scenes | 28.119 |
| synthesize_synopsis | 11.867 |
| make_embedding | 5.099 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 54.328 |
| branch_yolo_total | 13.244 |
| branch_audio_total | 82.511 |
