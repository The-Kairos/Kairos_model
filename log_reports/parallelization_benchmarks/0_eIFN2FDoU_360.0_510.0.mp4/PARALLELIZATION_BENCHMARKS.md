# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 20:53:50 UTC | 0_eIFN2FDoU_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.061 | - | - | - | - | - | - |
| 2026-06-22 12:33:56 UTC | 0_eIFN2FDoU_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 119.502 | 1.485 | 36.573 | 9.391 | 8.725 | 32.379 | 2.085 |

## 2026-06-21 20:53:50 UTC | 0_eIFN2FDoU_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0_eIFN2FDoU_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `0.061` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | - |
| save_clips | - |
| sample_frames | - |
| caption_frames | - |
| sample_fps | - |
| detect_object_yolo | - |
| audio_scan | - |
| asr_timings | - |
| ast_timings | - |
| describe_scenes | - |
| summarize_scenes | - |
| synthesize_synopsis | - |
| make_embedding | - |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | - |
| branch_yolo_total | - |
| branch_audio_total | - |

## 2026-06-22 12:33:56 UTC | 0_eIFN2FDoU_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0_eIFN2FDoU_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `119.502` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.485 |
| save_clips | - |
| sample_frames | 0.444 |
| caption_frames | 18.747 |
| sample_fps | 1.885 |
| detect_object_yolo | 6.408 |
| audio_scan | 13.776 |
| asr_timings | 9.916 |
| ast_timings | 12.872 |
| describe_scenes | 9.391 |
| summarize_scenes | 8.725 |
| synthesize_synopsis | 32.379 |
| make_embedding | 2.085 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 19.196 |
| branch_yolo_total | 8.299 |
| branch_audio_total | 36.573 |
