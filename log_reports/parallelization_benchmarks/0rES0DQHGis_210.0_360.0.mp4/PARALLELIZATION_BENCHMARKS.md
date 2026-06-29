# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 20:53:59 UTC | 0rES0DQHGis_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.060 | - | - | - | - | - | - |
| 2026-06-22 12:55:51 UTC | 0rES0DQHGis_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 220.833 | 0.761 | 50.880 | 14.491 | 50.249 | 49.762 | 3.341 |

## 2026-06-21 20:53:59 UTC | 0rES0DQHGis_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0rES0DQHGis_210.0_360.0.mp4`
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

## 2026-06-22 12:55:51 UTC | 0rES0DQHGis_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0rES0DQHGis_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `220.833` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.761 |
| save_clips | - |
| sample_frames | 1.038 |
| caption_frames | 38.124 |
| sample_fps | 2.167 |
| detect_object_yolo | 8.620 |
| audio_scan | 12.747 |
| asr_timings | 11.301 |
| ast_timings | 26.823 |
| describe_scenes | 14.491 |
| summarize_scenes | 50.249 |
| synthesize_synopsis | 49.762 |
| make_embedding | 3.341 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.167 |
| branch_yolo_total | 10.793 |
| branch_audio_total | 50.880 |
