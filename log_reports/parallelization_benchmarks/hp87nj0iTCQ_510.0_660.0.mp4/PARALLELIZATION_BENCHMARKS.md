# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 07:00:11 UTC | hp87nj0iTCQ_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 202.926 | 0.785 | 62.755 | 20.351 | 25.538 | 20.255 | 4.874 |

## 2026-06-26 07:00:11 UTC | hp87nj0iTCQ_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/hp87nj0iTCQ_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `202.926` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.785 |
| save_clips | - |
| sample_frames | 1.465 |
| caption_frames | 52.921 |
| sample_fps | 2.422 |
| detect_object_yolo | 10.159 |
| audio_scan | 15.122 |
| asr_timings | 9.334 |
| ast_timings | 38.290 |
| describe_scenes | 20.351 |
| summarize_scenes | 25.538 |
| synthesize_synopsis | 20.255 |
| make_embedding | 4.874 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 54.392 |
| branch_yolo_total | 12.587 |
| branch_audio_total | 62.755 |
