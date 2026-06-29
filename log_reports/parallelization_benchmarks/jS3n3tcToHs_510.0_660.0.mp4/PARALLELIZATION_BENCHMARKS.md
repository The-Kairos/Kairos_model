# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 11:00:53 UTC | jS3n3tcToHs_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 137.279 | 0.803 | 44.254 | 14.436 | 19.586 | 15.191 | 2.819 |

## 2026-06-26 11:00:53 UTC | jS3n3tcToHs_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jS3n3tcToHs_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `137.279` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.803 |
| save_clips | - |
| sample_frames | 0.780 |
| caption_frames | 27.632 |
| sample_fps | 2.134 |
| detect_object_yolo | 8.225 |
| audio_scan | 12.890 |
| asr_timings | 10.844 |
| ast_timings | 20.506 |
| describe_scenes | 14.436 |
| summarize_scenes | 19.586 |
| synthesize_synopsis | 15.191 |
| make_embedding | 2.819 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 28.418 |
| branch_yolo_total | 10.365 |
| branch_audio_total | 44.254 |
