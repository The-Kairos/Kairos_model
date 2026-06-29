# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 01:12:22 UTC | bWQ_wOvZ5Qo_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 155.882 | 0.811 | 57.259 | 8.900 | 16.061 | 10.677 | 3.839 |

## 2026-06-26 01:12:22 UTC | bWQ_wOvZ5Qo_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/bWQ_wOvZ5Qo_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `155.882` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.811 |
| save_clips | - |
| sample_frames | 1.481 |
| caption_frames | 43.985 |
| sample_fps | 2.414 |
| detect_object_yolo | 8.990 |
| audio_scan | 14.992 |
| asr_timings | 10.351 |
| ast_timings | 31.908 |
| describe_scenes | 8.900 |
| summarize_scenes | 16.061 |
| synthesize_synopsis | 10.677 |
| make_embedding | 3.839 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.471 |
| branch_yolo_total | 11.410 |
| branch_audio_total | 57.259 |
