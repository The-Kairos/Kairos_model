# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 13:15:59 UTC | 0H2xFsZtjNY_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 160.106 | 0.822 | 83.368 | 11.067 | 7.158 | 5.117 | 3.333 |

## 2026-06-27 13:15:59 UTC | 0H2xFsZtjNY_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0H2xFsZtjNY_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `160.106` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.822 |
| save_clips | - |
| sample_frames | 1.360 |
| caption_frames | 35.391 |
| sample_fps | 2.333 |
| detect_object_yolo | 8.738 |
| audio_scan | 15.970 |
| asr_timings | 40.411 |
| ast_timings | 26.978 |
| describe_scenes | 11.067 |
| summarize_scenes | 7.158 |
| synthesize_synopsis | 5.117 |
| make_embedding | 3.333 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.757 |
| branch_yolo_total | 11.077 |
| branch_audio_total | 83.368 |
