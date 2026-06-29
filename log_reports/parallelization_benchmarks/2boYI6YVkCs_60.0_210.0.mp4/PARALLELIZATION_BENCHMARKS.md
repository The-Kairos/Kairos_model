# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 09:49:28 UTC | 2boYI6YVkCs_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 167.651 | 1.544 | 65.550 | 9.920 | 9.030 | 6.727 | 4.778 |
| 2026-06-21 21:28:44 UTC | 2boYI6YVkCs_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 169.725 | 1.591 | 66.000 | 9.755 | 9.473 | 7.240 | 4.730 |

## 2026-06-21 09:49:28 UTC | 2boYI6YVkCs_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2boYI6YVkCs_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `167.651` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.544 |
| save_clips | - |
| sample_frames | 2.451 |
| caption_frames | 50.537 |
| sample_fps | 5.989 |
| detect_object_yolo | 9.800 |
| audio_scan | 12.726 |
| asr_timings | 15.123 |
| ast_timings | 37.692 |
| describe_scenes | 9.920 |
| summarize_scenes | 9.030 |
| synthesize_synopsis | 6.727 |
| make_embedding | 4.778 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 52.993 |
| branch_yolo_total | 15.795 |
| branch_audio_total | 65.550 |

## 2026-06-21 21:28:44 UTC | 2boYI6YVkCs_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2boYI6YVkCs_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `169.725` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.591 |
| save_clips | - |
| sample_frames | 2.507 |
| caption_frames | 50.612 |
| sample_fps | 6.131 |
| detect_object_yolo | 10.284 |
| audio_scan | 12.860 |
| asr_timings | 15.060 |
| ast_timings | 38.072 |
| describe_scenes | 9.755 |
| summarize_scenes | 9.473 |
| synthesize_synopsis | 7.240 |
| make_embedding | 4.730 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 53.125 |
| branch_yolo_total | 16.421 |
| branch_audio_total | 66.000 |
