# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 21:41:37 UTC | XXDq63KUBNo_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 158.829 | 0.793 | 59.393 | 13.543 | 17.533 | 12.929 | 3.509 |

## 2026-06-25 21:41:37 UTC | XXDq63KUBNo_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/XXDq63KUBNo_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `158.829` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.793 |
| save_clips | - |
| sample_frames | 1.132 |
| caption_frames | 38.186 |
| sample_fps | 2.223 |
| detect_object_yolo | 8.174 |
| audio_scan | 16.299 |
| asr_timings | 14.071 |
| ast_timings | 29.015 |
| describe_scenes | 13.543 |
| summarize_scenes | 17.533 |
| synthesize_synopsis | 12.929 |
| make_embedding | 3.509 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.324 |
| branch_yolo_total | 10.403 |
| branch_audio_total | 59.393 |
