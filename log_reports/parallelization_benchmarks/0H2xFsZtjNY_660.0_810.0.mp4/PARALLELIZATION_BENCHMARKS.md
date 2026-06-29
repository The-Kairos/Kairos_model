# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 13:23:17 UTC | 0H2xFsZtjNY_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 99.043 | 1.534 | 49.148 | 4.773 | 3.575 | 9.072 | 2.046 |

## 2026-06-27 13:23:17 UTC | 0H2xFsZtjNY_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0H2xFsZtjNY_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `99.043` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.534 |
| save_clips | - |
| sample_frames | 0.452 |
| caption_frames | 19.040 |
| sample_fps | 1.920 |
| detect_object_yolo | 6.077 |
| audio_scan | 14.859 |
| asr_timings | 21.311 |
| ast_timings | 12.969 |
| describe_scenes | 4.773 |
| summarize_scenes | 3.575 |
| synthesize_synopsis | 9.072 |
| make_embedding | 2.046 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 19.498 |
| branch_yolo_total | 8.002 |
| branch_audio_total | 49.148 |
