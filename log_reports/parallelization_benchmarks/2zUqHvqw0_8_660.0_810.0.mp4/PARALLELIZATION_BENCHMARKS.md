# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 15:21:28 UTC | 2zUqHvqw0_8_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 235.875 | 0.643 | 64.507 | 26.075 | 52.378 | 21.012 | 4.775 |
| 2026-06-24 09:20:12 UTC | 2zUqHvqw0_8_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 199.844 | 0.656 | 63.682 | 18.820 | 20.920 | 24.404 | 4.666 |

## 2026-06-23 15:21:28 UTC | 2zUqHvqw0_8_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2zUqHvqw0_8_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `235.875` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.643 |
| save_clips | - |
| sample_frames | 1.483 |
| caption_frames | 51.001 |
| sample_fps | 2.294 |
| detect_object_yolo | 10.261 |
| audio_scan | 16.117 |
| asr_timings | 10.483 |
| ast_timings | 37.899 |
| describe_scenes | 26.075 |
| summarize_scenes | 52.378 |
| synthesize_synopsis | 21.012 |
| make_embedding | 4.775 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 52.490 |
| branch_yolo_total | 12.560 |
| branch_audio_total | 64.507 |

## 2026-06-24 09:20:12 UTC | 2zUqHvqw0_8_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2zUqHvqw0_8_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `199.844` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.656 |
| save_clips | - |
| sample_frames | 1.497 |
| caption_frames | 51.527 |
| sample_fps | 2.273 |
| detect_object_yolo | 10.001 |
| audio_scan | 15.942 |
| asr_timings | 9.752 |
| ast_timings | 37.980 |
| describe_scenes | 18.820 |
| summarize_scenes | 20.920 |
| synthesize_synopsis | 24.404 |
| make_embedding | 4.666 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 53.031 |
| branch_yolo_total | 12.280 |
| branch_audio_total | 63.682 |
