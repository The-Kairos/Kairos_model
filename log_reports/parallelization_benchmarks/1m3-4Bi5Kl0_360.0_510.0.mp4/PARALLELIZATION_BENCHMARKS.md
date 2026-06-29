# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 14:08:17 UTC | 1m3-4Bi5Kl0_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 244.473 | 0.799 | 64.978 | 48.712 | 23.060 | 29.344 | 5.824 |
| 2026-06-27 15:30:02 UTC | 1m3-4Bi5Kl0_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 178.498 | 0.825 | 69.091 | 12.975 | 7.861 | 8.390 | 5.344 |

## 2026-06-23 14:08:17 UTC | 1m3-4Bi5Kl0_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1m3-4Bi5Kl0_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `244.473` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.799 |
| save_clips | - |
| sample_frames | 1.675 |
| caption_frames | 55.500 |
| sample_fps | 2.594 |
| detect_object_yolo | 10.626 |
| audio_scan | 13.758 |
| asr_timings | 7.952 |
| ast_timings | 43.260 |
| describe_scenes | 48.712 |
| summarize_scenes | 23.060 |
| synthesize_synopsis | 29.344 |
| make_embedding | 5.824 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 57.180 |
| branch_yolo_total | 13.225 |
| branch_audio_total | 64.978 |

## 2026-06-27 15:30:02 UTC | 1m3-4Bi5Kl0_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1m3-4Bi5Kl0_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `178.498` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.825 |
| save_clips | - |
| sample_frames | 1.688 |
| caption_frames | 57.335 |
| sample_fps | 2.643 |
| detect_object_yolo | 10.923 |
| audio_scan | 13.968 |
| asr_timings | 11.163 |
| ast_timings | 43.952 |
| describe_scenes | 12.975 |
| summarize_scenes | 7.861 |
| synthesize_synopsis | 8.390 |
| make_embedding | 5.344 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 59.029 |
| branch_yolo_total | 13.572 |
| branch_audio_total | 69.091 |
