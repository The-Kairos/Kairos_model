# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 20:53:53 UTC | 0_eIFN2FDoU_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.060 | - | - | - | - | - | - |
| 2026-06-22 12:37:08 UTC | 0_eIFN2FDoU_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 107.931 | 1.491 | 34.529 | 5.574 | 16.148 | 19.503 | 2.126 |

## 2026-06-21 20:53:53 UTC | 0_eIFN2FDoU_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0_eIFN2FDoU_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `0.060` sec

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

## 2026-06-22 12:37:08 UTC | 0_eIFN2FDoU_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0_eIFN2FDoU_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `107.931` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.491 |
| save_clips | - |
| sample_frames | 0.452 |
| caption_frames | 18.387 |
| sample_fps | 1.903 |
| detect_object_yolo | 6.437 |
| audio_scan | 10.641 |
| asr_timings | 10.834 |
| ast_timings | 13.045 |
| describe_scenes | 5.574 |
| summarize_scenes | 16.148 |
| synthesize_synopsis | 19.503 |
| make_embedding | 2.126 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 18.844 |
| branch_yolo_total | 8.345 |
| branch_audio_total | 34.529 |
