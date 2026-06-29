# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 10:20:16 UTC | MxM1yP8cRD4_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 135.451 | 0.779 | 42.899 | 13.222 | 22.131 | 15.552 | 2.539 |

## 2026-06-25 10:20:16 UTC | MxM1yP8cRD4_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/MxM1yP8cRD4_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `135.451` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.779 |
| save_clips | - |
| sample_frames | 0.702 |
| caption_frames | 27.068 |
| sample_fps | 2.039 |
| detect_object_yolo | 7.101 |
| audio_scan | 12.762 |
| asr_timings | 11.216 |
| ast_timings | 18.912 |
| describe_scenes | 13.222 |
| summarize_scenes | 22.131 |
| synthesize_synopsis | 15.552 |
| make_embedding | 2.539 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 27.776 |
| branch_yolo_total | 9.146 |
| branch_audio_total | 42.899 |
