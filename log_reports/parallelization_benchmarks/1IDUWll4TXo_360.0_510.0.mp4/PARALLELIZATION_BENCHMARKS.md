# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 13:20:19 UTC | 1IDUWll4TXo_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 238.750 | 0.777 | 59.785 | 33.448 | 29.726 | 40.308 | 5.043 |
| 2026-06-27 14:57:07 UTC | 1IDUWll4TXo_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 162.638 | 0.760 | 59.664 | 12.137 | 10.189 | 5.614 | 5.076 |

## 2026-06-23 13:20:19 UTC | 1IDUWll4TXo_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1IDUWll4TXo_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `238.750` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.777 |
| save_clips | - |
| sample_frames | 1.828 |
| caption_frames | 54.193 |
| sample_fps | 2.516 |
| detect_object_yolo | 9.743 |
| audio_scan | 11.571 |
| asr_timings | 7.717 |
| ast_timings | 40.488 |
| describe_scenes | 33.448 |
| summarize_scenes | 29.726 |
| synthesize_synopsis | 40.308 |
| make_embedding | 5.043 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 56.027 |
| branch_yolo_total | 12.264 |
| branch_audio_total | 59.785 |

## 2026-06-27 14:57:07 UTC | 1IDUWll4TXo_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1IDUWll4TXo_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `162.638` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.760 |
| save_clips | - |
| sample_frames | 1.841 |
| caption_frames | 53.503 |
| sample_fps | 2.530 |
| detect_object_yolo | 9.919 |
| audio_scan | 11.844 |
| asr_timings | 6.802 |
| ast_timings | 41.010 |
| describe_scenes | 12.137 |
| summarize_scenes | 10.189 |
| synthesize_synopsis | 5.614 |
| make_embedding | 5.076 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 55.350 |
| branch_yolo_total | 12.455 |
| branch_audio_total | 59.664 |
