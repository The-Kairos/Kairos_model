# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 08:18:13 UTC | iCbjk3uRtqA_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 198.593 | 0.790 | 61.246 | 29.352 | 16.567 | 18.899 | 4.686 |

## 2026-06-26 08:18:13 UTC | iCbjk3uRtqA_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/iCbjk3uRtqA_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `198.593` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.790 |
| save_clips | - |
| sample_frames | 1.380 |
| caption_frames | 51.758 |
| sample_fps | 2.365 |
| detect_object_yolo | 10.134 |
| audio_scan | 12.960 |
| asr_timings | 10.093 |
| ast_timings | 38.185 |
| describe_scenes | 29.352 |
| summarize_scenes | 16.567 |
| synthesize_synopsis | 18.899 |
| make_embedding | 4.686 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 53.145 |
| branch_yolo_total | 12.505 |
| branch_audio_total | 61.246 |
