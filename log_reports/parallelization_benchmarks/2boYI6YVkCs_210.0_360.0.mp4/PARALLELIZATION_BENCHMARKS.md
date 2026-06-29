# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 09:46:39 UTC | 2boYI6YVkCs_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 136.308 | 1.582 | 56.391 | 7.977 | 6.219 | 8.642 | 3.422 |
| 2026-06-21 21:25:53 UTC | 2boYI6YVkCs_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 147.268 | 1.632 | 60.485 | 9.284 | 12.294 | 6.975 | 3.310 |

## 2026-06-21 09:46:39 UTC | 2boYI6YVkCs_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2boYI6YVkCs_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `136.308` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.582 |
| save_clips | - |
| sample_frames | 2.259 |
| caption_frames | 34.038 |
| sample_fps | 6.023 |
| detect_object_yolo | 8.432 |
| audio_scan | 13.767 |
| asr_timings | 16.463 |
| ast_timings | 26.153 |
| describe_scenes | 7.977 |
| summarize_scenes | 6.219 |
| synthesize_synopsis | 8.642 |
| make_embedding | 3.422 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.303 |
| branch_yolo_total | 14.461 |
| branch_audio_total | 56.391 |

## 2026-06-21 21:25:53 UTC | 2boYI6YVkCs_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2boYI6YVkCs_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `147.268` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.632 |
| save_clips | - |
| sample_frames | 2.254 |
| caption_frames | 34.807 |
| sample_fps | 6.110 |
| detect_object_yolo | 8.729 |
| audio_scan | 13.965 |
| asr_timings | 20.111 |
| ast_timings | 26.401 |
| describe_scenes | 9.284 |
| summarize_scenes | 12.294 |
| synthesize_synopsis | 6.975 |
| make_embedding | 3.310 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.067 |
| branch_yolo_total | 14.844 |
| branch_audio_total | 60.485 |
