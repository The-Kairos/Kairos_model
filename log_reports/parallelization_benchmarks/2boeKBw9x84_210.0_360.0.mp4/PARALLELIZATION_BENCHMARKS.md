# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 14:50:38 UTC | 2boeKBw9x84_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 156.651 | 0.674 | 50.146 | 23.343 | 10.078 | 21.768 | 3.294 |
| 2026-06-24 08:51:46 UTC | 2boeKBw9x84_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 160.859 | 0.705 | 49.985 | 23.124 | 16.512 | 19.483 | 3.375 |

## 2026-06-23 14:50:38 UTC | 2boeKBw9x84_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2boeKBw9x84_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `156.651` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.674 |
| save_clips | - |
| sample_frames | 1.258 |
| caption_frames | 34.277 |
| sample_fps | 2.139 |
| detect_object_yolo | 8.302 |
| audio_scan | 14.772 |
| asr_timings | 8.836 |
| ast_timings | 26.530 |
| describe_scenes | 23.343 |
| summarize_scenes | 10.078 |
| synthesize_synopsis | 21.768 |
| make_embedding | 3.294 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 35.541 |
| branch_yolo_total | 10.446 |
| branch_audio_total | 50.146 |

## 2026-06-24 08:51:46 UTC | 2boeKBw9x84_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2boeKBw9x84_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `160.859` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.705 |
| save_clips | - |
| sample_frames | 1.263 |
| caption_frames | 34.354 |
| sample_fps | 2.165 |
| detect_object_yolo | 8.480 |
| audio_scan | 14.869 |
| asr_timings | 8.521 |
| ast_timings | 26.587 |
| describe_scenes | 23.124 |
| summarize_scenes | 16.512 |
| synthesize_synopsis | 19.483 |
| make_embedding | 3.375 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 35.622 |
| branch_yolo_total | 10.651 |
| branch_audio_total | 49.985 |
