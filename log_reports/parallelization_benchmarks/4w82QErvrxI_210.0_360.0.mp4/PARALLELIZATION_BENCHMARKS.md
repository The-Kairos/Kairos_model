# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 17:25:46 UTC | 4w82QErvrxI_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 78.965 | 0.764 | 23.500 | 8.058 | 11.052 | 16.676 | 1.320 |
| 2026-06-24 11:18:25 UTC | 4w82QErvrxI_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 82.167 | 0.812 | 23.546 | 10.028 | 13.706 | 14.990 | 1.362 |

## 2026-06-23 17:25:46 UTC | 4w82QErvrxI_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/4w82QErvrxI_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `78.965` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.764 |
| save_clips | - |
| sample_frames | 0.113 |
| caption_frames | 8.548 |
| sample_fps | 1.714 |
| detect_object_yolo | 5.861 |
| audio_scan | 6.363 |
| asr_timings | 12.886 |
| ast_timings | 4.243 |
| describe_scenes | 8.058 |
| summarize_scenes | 11.052 |
| synthesize_synopsis | 16.676 |
| make_embedding | 1.320 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 8.667 |
| branch_yolo_total | 7.581 |
| branch_audio_total | 23.500 |

## 2026-06-24 11:18:25 UTC | 4w82QErvrxI_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/4w82QErvrxI_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `82.167` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.812 |
| save_clips | - |
| sample_frames | 0.113 |
| caption_frames | 8.592 |
| sample_fps | 1.745 |
| detect_object_yolo | 5.889 |
| audio_scan | 6.470 |
| asr_timings | 12.705 |
| ast_timings | 4.362 |
| describe_scenes | 10.028 |
| summarize_scenes | 13.706 |
| synthesize_synopsis | 14.990 |
| make_embedding | 1.362 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 8.711 |
| branch_yolo_total | 7.641 |
| branch_audio_total | 23.546 |
