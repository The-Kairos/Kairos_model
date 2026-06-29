# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 14:04:11 UTC | 1m3-4Bi5Kl0_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 262.692 | 0.794 | 66.427 | 56.752 | 22.532 | 33.427 | 11.129 |
| 2026-06-27 15:27:03 UTC | 1m3-4Bi5Kl0_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 175.277 | 0.810 | 66.274 | 14.386 | 7.850 | 6.822 | 5.325 |

## 2026-06-23 14:04:11 UTC | 1m3-4Bi5Kl0_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1m3-4Bi5Kl0_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `262.692` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.794 |
| save_clips | - |
| sample_frames | 1.483 |
| caption_frames | 55.537 |
| sample_fps | 2.568 |
| detect_object_yolo | 10.666 |
| audio_scan | 12.729 |
| asr_timings | 10.068 |
| ast_timings | 43.622 |
| describe_scenes | 56.752 |
| summarize_scenes | 22.532 |
| synthesize_synopsis | 33.427 |
| make_embedding | 11.129 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 57.026 |
| branch_yolo_total | 13.240 |
| branch_audio_total | 66.427 |

## 2026-06-27 15:27:03 UTC | 1m3-4Bi5Kl0_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1m3-4Bi5Kl0_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `175.277` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.810 |
| save_clips | - |
| sample_frames | 1.489 |
| caption_frames | 57.445 |
| sample_fps | 2.554 |
| detect_object_yolo | 10.902 |
| audio_scan | 12.878 |
| asr_timings | 9.276 |
| ast_timings | 44.111 |
| describe_scenes | 14.386 |
| summarize_scenes | 7.850 |
| synthesize_synopsis | 6.822 |
| make_embedding | 5.325 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 58.939 |
| branch_yolo_total | 13.462 |
| branch_audio_total | 66.274 |
