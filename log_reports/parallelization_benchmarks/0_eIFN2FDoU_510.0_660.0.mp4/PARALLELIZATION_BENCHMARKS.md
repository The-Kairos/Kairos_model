# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 20:53:52 UTC | 0_eIFN2FDoU_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.060 | - | - | - | - | - | - |
| 2026-06-22 12:35:19 UTC | 0_eIFN2FDoU_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 81.465 | 0.757 | 30.513 | 7.407 | 4.415 | 17.355 | 1.525 |

## 2026-06-21 20:53:52 UTC | 0_eIFN2FDoU_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0_eIFN2FDoU_510.0_660.0.mp4`
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

## 2026-06-22 12:35:19 UTC | 0_eIFN2FDoU_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0_eIFN2FDoU_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `81.465` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.757 |
| save_clips | - |
| sample_frames | 0.130 |
| caption_frames | 10.240 |
| sample_fps | 1.765 |
| detect_object_yolo | 5.994 |
| audio_scan | 12.752 |
| asr_timings | 10.686 |
| ast_timings | 7.066 |
| describe_scenes | 7.407 |
| summarize_scenes | 4.415 |
| synthesize_synopsis | 17.355 |
| make_embedding | 1.525 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 10.376 |
| branch_yolo_total | 7.764 |
| branch_audio_total | 30.513 |
