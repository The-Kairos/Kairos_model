# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 14:38:33 UTC | 2YGUY_4d8D0_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 205.634 | 0.780 | 56.433 | 23.906 | 19.713 | 33.733 | 4.503 |
| 2026-06-27 15:52:45 UTC | 2YGUY_4d8D0_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 155.666 | 0.820 | 57.081 | 8.876 | 7.661 | 9.998 | 4.636 |

## 2026-06-23 14:38:33 UTC | 2YGUY_4d8D0_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2YGUY_4d8D0_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `205.634` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.780 |
| save_clips | - |
| sample_frames | 1.331 |
| caption_frames | 51.154 |
| sample_fps | 2.485 |
| detect_object_yolo | 10.213 |
| audio_scan | 6.409 |
| asr_timings | 13.040 |
| ast_timings | 36.976 |
| describe_scenes | 23.906 |
| summarize_scenes | 19.713 |
| synthesize_synopsis | 33.733 |
| make_embedding | 4.503 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 52.491 |
| branch_yolo_total | 12.704 |
| branch_audio_total | 56.433 |

## 2026-06-27 15:52:45 UTC | 2YGUY_4d8D0_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2YGUY_4d8D0_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `155.666` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.820 |
| save_clips | - |
| sample_frames | 1.344 |
| caption_frames | 51.004 |
| sample_fps | 2.477 |
| detect_object_yolo | 10.344 |
| audio_scan | 6.466 |
| asr_timings | 13.330 |
| ast_timings | 37.277 |
| describe_scenes | 8.876 |
| summarize_scenes | 7.661 |
| synthesize_synopsis | 9.998 |
| make_embedding | 4.636 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 52.355 |
| branch_yolo_total | 12.828 |
| branch_audio_total | 57.081 |
