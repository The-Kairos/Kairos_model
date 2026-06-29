# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 12:47:36 UTC | k4LLzwmwJS8_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 201.085 | 0.835 | 64.022 | 21.110 | 17.611 | 20.302 | 5.091 |

## 2026-06-26 12:47:36 UTC | k4LLzwmwJS8_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/k4LLzwmwJS8_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `201.085` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.835 |
| save_clips | - |
| sample_frames | 1.538 |
| caption_frames | 55.109 |
| sample_fps | 2.572 |
| detect_object_yolo | 11.408 |
| audio_scan | 11.992 |
| asr_timings | 10.435 |
| ast_timings | 41.587 |
| describe_scenes | 21.110 |
| summarize_scenes | 17.611 |
| synthesize_synopsis | 20.302 |
| make_embedding | 5.091 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 56.653 |
| branch_yolo_total | 13.986 |
| branch_audio_total | 64.022 |
