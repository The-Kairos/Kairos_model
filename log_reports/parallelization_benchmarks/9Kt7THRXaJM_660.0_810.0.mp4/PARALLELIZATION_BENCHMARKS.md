# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 17:53:49 UTC | 9Kt7THRXaJM_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 150.580 | 0.884 | 59.416 | 13.786 | 11.947 | 11.610 | 3.316 |

## 2026-06-24 17:53:49 UTC | 9Kt7THRXaJM_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/9Kt7THRXaJM_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `150.580` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.884 |
| save_clips | - |
| sample_frames | 1.107 |
| caption_frames | 36.664 |
| sample_fps | 2.250 |
| detect_object_yolo | 8.211 |
| audio_scan | 10.758 |
| asr_timings | 21.223 |
| ast_timings | 27.426 |
| describe_scenes | 13.786 |
| summarize_scenes | 11.947 |
| synthesize_synopsis | 11.610 |
| make_embedding | 3.316 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.777 |
| branch_yolo_total | 10.467 |
| branch_audio_total | 59.416 |
