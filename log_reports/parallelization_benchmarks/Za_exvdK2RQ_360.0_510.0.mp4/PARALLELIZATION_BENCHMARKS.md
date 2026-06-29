# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 22:33:18 UTC | Za_exvdK2RQ_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 164.640 | 0.848 | 59.558 | 16.514 | 12.300 | 8.481 | 4.152 |

## 2026-06-25 22:33:18 UTC | Za_exvdK2RQ_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Za_exvdK2RQ_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `164.640` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.848 |
| save_clips | - |
| sample_frames | 1.702 |
| caption_frames | 47.517 |
| sample_fps | 2.616 |
| detect_object_yolo | 9.550 |
| audio_scan | 13.753 |
| asr_timings | 10.070 |
| ast_timings | 35.727 |
| describe_scenes | 16.514 |
| summarize_scenes | 12.300 |
| synthesize_synopsis | 8.481 |
| make_embedding | 4.152 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.226 |
| branch_yolo_total | 12.171 |
| branch_audio_total | 59.558 |
