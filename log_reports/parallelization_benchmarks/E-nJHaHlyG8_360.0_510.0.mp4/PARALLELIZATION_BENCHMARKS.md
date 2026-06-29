# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 23:30:42 UTC | E-nJHaHlyG8_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 173.217 | 0.920 | 61.091 | 13.098 | 13.534 | 6.518 | 5.033 |

## 2026-06-24 23:30:42 UTC | E-nJHaHlyG8_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/E-nJHaHlyG8_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `173.217` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.920 |
| save_clips | - |
| sample_frames | 1.464 |
| caption_frames | 56.375 |
| sample_fps | 2.567 |
| detect_object_yolo | 11.165 |
| audio_scan | 11.785 |
| asr_timings | 8.914 |
| ast_timings | 40.383 |
| describe_scenes | 13.098 |
| summarize_scenes | 13.534 |
| synthesize_synopsis | 6.518 |
| make_embedding | 5.033 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 57.845 |
| branch_yolo_total | 13.738 |
| branch_audio_total | 61.091 |
