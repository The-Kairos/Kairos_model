# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 00:13:44 UTC | u-UA8t2EVpA_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 222.435 | 0.667 | 76.639 | 22.673 | 16.415 | 7.672 | 6.511 |

## 2026-06-27 00:13:44 UTC | u-UA8t2EVpA_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/u-UA8t2EVpA_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `222.435` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.667 |
| save_clips | - |
| sample_frames | 1.807 |
| caption_frames | 73.047 |
| sample_fps | 2.563 |
| detect_object_yolo | 12.977 |
| audio_scan | 9.668 |
| asr_timings | 12.510 |
| ast_timings | 54.453 |
| describe_scenes | 22.673 |
| summarize_scenes | 16.415 |
| synthesize_synopsis | 7.672 |
| make_embedding | 6.511 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 74.860 |
| branch_yolo_total | 15.545 |
| branch_audio_total | 76.639 |
