# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 16:05:54 UTC | 3LSElHQKRho_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 249.933 | 0.801 | 74.635 | 33.087 | 21.110 | 23.451 | 6.613 |
| 2026-06-24 10:02:43 UTC | 3LSElHQKRho_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 240.419 | 0.812 | 74.353 | 30.348 | 23.640 | 16.190 | 6.454 |

## 2026-06-23 16:05:54 UTC | 3LSElHQKRho_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3LSElHQKRho_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `249.933` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.801 |
| save_clips | - |
| sample_frames | 1.811 |
| caption_frames | 71.366 |
| sample_fps | 2.796 |
| detect_object_yolo | 12.839 |
| audio_scan | 10.667 |
| asr_timings | 11.913 |
| ast_timings | 52.046 |
| describe_scenes | 33.087 |
| summarize_scenes | 21.110 |
| synthesize_synopsis | 23.451 |
| make_embedding | 6.613 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 73.183 |
| branch_yolo_total | 15.640 |
| branch_audio_total | 74.635 |

## 2026-06-24 10:02:43 UTC | 3LSElHQKRho_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3LSElHQKRho_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `240.419` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.812 |
| save_clips | - |
| sample_frames | 1.833 |
| caption_frames | 69.984 |
| sample_fps | 2.800 |
| detect_object_yolo | 12.594 |
| audio_scan | 10.702 |
| asr_timings | 11.549 |
| ast_timings | 52.093 |
| describe_scenes | 30.348 |
| summarize_scenes | 23.640 |
| synthesize_synopsis | 16.190 |
| make_embedding | 6.454 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 71.823 |
| branch_yolo_total | 15.400 |
| branch_audio_total | 74.353 |
