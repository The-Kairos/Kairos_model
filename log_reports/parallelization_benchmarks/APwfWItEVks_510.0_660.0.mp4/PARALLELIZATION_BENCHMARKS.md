# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 18:52:24 UTC | APwfWItEVks_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 127.969 | 0.769 | 42.735 | 8.613 | 6.621 | 29.133 | 2.551 |

## 2026-06-24 18:52:24 UTC | APwfWItEVks_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/APwfWItEVks_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `127.969` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.769 |
| save_clips | - |
| sample_frames | 0.574 |
| caption_frames | 26.274 |
| sample_fps | 2.008 |
| detect_object_yolo | 7.288 |
| audio_scan | 11.916 |
| asr_timings | 12.161 |
| ast_timings | 18.649 |
| describe_scenes | 8.613 |
| summarize_scenes | 6.621 |
| synthesize_synopsis | 29.133 |
| make_embedding | 2.551 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 26.854 |
| branch_yolo_total | 9.302 |
| branch_audio_total | 42.735 |
