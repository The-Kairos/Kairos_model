# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 15:42:17 UTC | QrFLjLZIeig_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 181.441 | 0.762 | 57.150 | 21.466 | 20.542 | 21.558 | 3.883 |

## 2026-06-25 15:42:17 UTC | QrFLjLZIeig_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/QrFLjLZIeig_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `181.441` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.762 |
| save_clips | - |
| sample_frames | 1.068 |
| caption_frames | 42.535 |
| sample_fps | 2.255 |
| detect_object_yolo | 8.801 |
| audio_scan | 16.646 |
| asr_timings | 8.612 |
| ast_timings | 31.884 |
| describe_scenes | 21.466 |
| summarize_scenes | 20.542 |
| synthesize_synopsis | 21.558 |
| make_embedding | 3.883 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.609 |
| branch_yolo_total | 11.061 |
| branch_audio_total | 57.150 |
