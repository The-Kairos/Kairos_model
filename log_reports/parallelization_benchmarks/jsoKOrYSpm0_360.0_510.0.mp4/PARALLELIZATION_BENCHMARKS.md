# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 12:27:00 UTC | jsoKOrYSpm0_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 222.932 | 0.801 | 62.955 | 20.582 | 58.121 | 16.925 | 4.152 |

## 2026-06-26 12:27:00 UTC | jsoKOrYSpm0_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jsoKOrYSpm0_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `222.932` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.801 |
| save_clips | - |
| sample_frames | 1.124 |
| caption_frames | 44.768 |
| sample_fps | 2.344 |
| detect_object_yolo | 9.732 |
| audio_scan | 15.089 |
| asr_timings | 12.240 |
| ast_timings | 35.619 |
| describe_scenes | 20.582 |
| summarize_scenes | 58.121 |
| synthesize_synopsis | 16.925 |
| make_embedding | 4.152 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.898 |
| branch_yolo_total | 12.082 |
| branch_audio_total | 62.955 |
