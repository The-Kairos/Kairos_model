# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 12:41:17 UTC | 13RZnGLj-iQ_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 123.313 | 0.767 | 37.576 | 14.475 | 8.551 | 23.284 | 2.535 |
| 2026-06-27 14:27:44 UTC | 13RZnGLj-iQ_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 96.558 | 0.781 | 36.563 | 6.623 | 7.790 | 4.669 | 2.648 |

## 2026-06-23 12:41:17 UTC | 13RZnGLj-iQ_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/13RZnGLj-iQ_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `123.313` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.767 |
| save_clips | - |
| sample_frames | 0.645 |
| caption_frames | 24.880 |
| sample_fps | 1.991 |
| detect_object_yolo | 7.203 |
| audio_scan | 6.408 |
| asr_timings | 12.996 |
| ast_timings | 18.163 |
| describe_scenes | 14.475 |
| summarize_scenes | 8.551 |
| synthesize_synopsis | 23.284 |
| make_embedding | 2.535 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 25.530 |
| branch_yolo_total | 9.200 |
| branch_audio_total | 37.576 |

## 2026-06-27 14:27:44 UTC | 13RZnGLj-iQ_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/13RZnGLj-iQ_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `96.558` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.781 |
| save_clips | - |
| sample_frames | 0.651 |
| caption_frames | 26.176 |
| sample_fps | 2.008 |
| detect_object_yolo | 7.254 |
| audio_scan | 6.452 |
| asr_timings | 11.741 |
| ast_timings | 18.362 |
| describe_scenes | 6.623 |
| summarize_scenes | 7.790 |
| synthesize_synopsis | 4.669 |
| make_embedding | 2.648 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 26.833 |
| branch_yolo_total | 9.268 |
| branch_audio_total | 36.563 |
