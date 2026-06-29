# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 12:08:36 UTC | 0kAcEdn-C1M_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 38.941 | 1.764 | 61.291 | 31.321 | 13.415 | 19.058 | 3.939 |
| 2026-06-27 14:02:02 UTC | 0kAcEdn-C1M_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 144.154 | 0.803 | 49.478 | 13.787 | 8.393 | 8.861 | 3.979 |

## 2026-06-23 12:08:36 UTC | 0kAcEdn-C1M_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0kAcEdn-C1M_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `38.941` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.764 |
| save_clips | - |
| sample_frames | 3.285 |
| caption_frames | 42.560 |
| sample_fps | 6.796 |
| detect_object_yolo | 9.616 |
| audio_scan | 11.782 |
| asr_timings | 17.016 |
| ast_timings | 32.485 |
| describe_scenes | 31.321 |
| summarize_scenes | 13.415 |
| synthesize_synopsis | 19.058 |
| make_embedding | 3.939 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.850 |
| branch_yolo_total | 16.418 |
| branch_audio_total | 61.291 |

## 2026-06-27 14:02:02 UTC | 0kAcEdn-C1M_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0kAcEdn-C1M_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `144.154` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.803 |
| save_clips | - |
| sample_frames | 1.337 |
| caption_frames | 44.554 |
| sample_fps | 2.409 |
| detect_object_yolo | 9.098 |
| audio_scan | 8.655 |
| asr_timings | 7.657 |
| ast_timings | 33.158 |
| describe_scenes | 13.787 |
| summarize_scenes | 8.393 |
| synthesize_synopsis | 8.861 |
| make_embedding | 3.979 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.898 |
| branch_yolo_total | 11.513 |
| branch_audio_total | 49.478 |
