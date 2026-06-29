# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 23:55:32 UTC | EtXnXxOegko_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 109.343 | 0.778 | 44.218 | 5.246 | 5.844 | 12.491 | 2.507 |

## 2026-06-24 23:55:32 UTC | EtXnXxOegko_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/EtXnXxOegko_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `109.343` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.778 |
| save_clips | - |
| sample_frames | 0.696 |
| caption_frames | 26.899 |
| sample_fps | 2.087 |
| detect_object_yolo | 7.164 |
| audio_scan | 14.883 |
| asr_timings | 10.562 |
| ast_timings | 18.765 |
| describe_scenes | 5.246 |
| summarize_scenes | 5.844 |
| synthesize_synopsis | 12.491 |
| make_embedding | 2.507 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 27.601 |
| branch_yolo_total | 9.257 |
| branch_audio_total | 44.218 |
