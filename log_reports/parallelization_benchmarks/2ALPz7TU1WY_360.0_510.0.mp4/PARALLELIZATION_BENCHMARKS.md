# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 14:17:11 UTC | 2ALPz7TU1WY_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 184.505 | 0.844 | 43.412 | 16.549 | 52.670 | 23.323 | 3.179 |
| 2026-06-27 15:36:15 UTC | 2ALPz7TU1WY_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 116.652 | 0.778 | 43.276 | 7.016 | 7.589 | 8.529 | 3.123 |

## 2026-06-23 14:17:11 UTC | 2ALPz7TU1WY_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2ALPz7TU1WY_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `184.505` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.844 |
| save_clips | - |
| sample_frames | 0.762 |
| caption_frames | 32.495 |
| sample_fps | 2.080 |
| detect_object_yolo | 7.805 |
| audio_scan | 9.481 |
| asr_timings | 9.636 |
| ast_timings | 24.287 |
| describe_scenes | 16.549 |
| summarize_scenes | 52.670 |
| synthesize_synopsis | 23.323 |
| make_embedding | 3.179 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 33.263 |
| branch_yolo_total | 9.890 |
| branch_audio_total | 43.412 |

## 2026-06-27 15:36:15 UTC | 2ALPz7TU1WY_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2ALPz7TU1WY_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `116.652` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.778 |
| save_clips | - |
| sample_frames | 0.778 |
| caption_frames | 34.057 |
| sample_fps | 2.104 |
| detect_object_yolo | 7.981 |
| audio_scan | 9.655 |
| asr_timings | 9.191 |
| ast_timings | 24.422 |
| describe_scenes | 7.016 |
| summarize_scenes | 7.589 |
| synthesize_synopsis | 8.529 |
| make_embedding | 3.123 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 34.840 |
| branch_yolo_total | 10.091 |
| branch_audio_total | 43.276 |
