# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 20:54:00 UTC | 0rES0DQHGis_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.061 | - | - | - | - | - | - |
| 2026-06-22 12:58:07 UTC | 0rES0DQHGis_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 134.272 | 0.772 | 41.996 | 10.672 | 18.669 | 21.327 | 2.857 |

## 2026-06-21 20:54:00 UTC | 0rES0DQHGis_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0rES0DQHGis_60.0_210.0.mp4`
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

## 2026-06-22 12:58:07 UTC | 0rES0DQHGis_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0rES0DQHGis_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `134.272` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.772 |
| save_clips | - |
| sample_frames | 0.561 |
| caption_frames | 27.054 |
| sample_fps | 1.949 |
| detect_object_yolo | 7.040 |
| audio_scan | 13.775 |
| asr_timings | 9.599 |
| ast_timings | 18.614 |
| describe_scenes | 10.672 |
| summarize_scenes | 18.669 |
| synthesize_synopsis | 21.327 |
| make_embedding | 2.857 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 27.621 |
| branch_yolo_total | 8.996 |
| branch_audio_total | 41.996 |
