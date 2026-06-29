# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 12:26:31 UTC | 0o3PxUAyhr4_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 243.255 | 0.781 | 69.026 | 42.992 | 23.499 | 17.389 | 6.048 |
| 2026-06-27 14:17:06 UTC | 0o3PxUAyhr4_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 205.190 | 0.786 | 70.001 | 18.392 | 13.076 | 12.418 | 6.140 |

## 2026-06-23 12:26:31 UTC | 0o3PxUAyhr4_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0o3PxUAyhr4_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `243.255` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.781 |
| save_clips | - |
| sample_frames | 1.532 |
| caption_frames | 66.268 |
| sample_fps | 2.543 |
| detect_object_yolo | 11.786 |
| audio_scan | 10.579 |
| asr_timings | 9.934 |
| ast_timings | 48.505 |
| describe_scenes | 42.992 |
| summarize_scenes | 23.499 |
| synthesize_synopsis | 17.389 |
| make_embedding | 6.048 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 67.806 |
| branch_yolo_total | 14.334 |
| branch_audio_total | 69.026 |

## 2026-06-27 14:17:06 UTC | 0o3PxUAyhr4_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0o3PxUAyhr4_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `205.190` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.786 |
| save_clips | - |
| sample_frames | 1.596 |
| caption_frames | 66.619 |
| sample_fps | 2.600 |
| detect_object_yolo | 12.134 |
| audio_scan | 10.741 |
| asr_timings | 9.968 |
| ast_timings | 49.283 |
| describe_scenes | 18.392 |
| summarize_scenes | 13.076 |
| synthesize_synopsis | 12.418 |
| make_embedding | 6.140 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 68.221 |
| branch_yolo_total | 14.740 |
| branch_audio_total | 70.001 |
