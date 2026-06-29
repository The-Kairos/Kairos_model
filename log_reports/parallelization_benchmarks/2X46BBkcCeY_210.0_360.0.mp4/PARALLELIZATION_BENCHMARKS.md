# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 09:38:35 UTC | 2X46BBkcCeY_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 145.535 | 1.959 | 48.512 | 10.236 | 13.925 | 7.143 | 3.615 |
| 2026-06-21 21:17:25 UTC | 2X46BBkcCeY_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 142.533 | 2.016 | 49.324 | 11.391 | 9.731 | 6.136 | 3.685 |

## 2026-06-21 09:38:35 UTC | 2X46BBkcCeY_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2X46BBkcCeY_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `145.535` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.959 |
| save_clips | - |
| sample_frames | 3.240 |
| caption_frames | 40.148 |
| sample_fps | 6.676 |
| detect_object_yolo | 8.743 |
| audio_scan | 8.594 |
| asr_timings | 10.460 |
| ast_timings | 29.449 |
| describe_scenes | 10.236 |
| summarize_scenes | 13.925 |
| synthesize_synopsis | 7.143 |
| make_embedding | 3.615 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.395 |
| branch_yolo_total | 15.425 |
| branch_audio_total | 48.512 |

## 2026-06-21 21:17:25 UTC | 2X46BBkcCeY_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2X46BBkcCeY_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `142.533` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.016 |
| save_clips | - |
| sample_frames | 3.278 |
| caption_frames | 39.773 |
| sample_fps | 6.755 |
| detect_object_yolo | 9.039 |
| audio_scan | 8.643 |
| asr_timings | 10.460 |
| ast_timings | 30.213 |
| describe_scenes | 11.391 |
| summarize_scenes | 9.731 |
| synthesize_synopsis | 6.136 |
| make_embedding | 3.685 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.057 |
| branch_yolo_total | 15.800 |
| branch_audio_total | 49.324 |
