# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 12:13:26 UTC | 0kAcEdn-C1M_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 149.537 | 0.744 | 46.092 | 24.579 | 10.073 | 20.350 | 3.044 |
| 2026-06-27 14:06:22 UTC | 0kAcEdn-C1M_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 122.639 | 0.779 | 46.702 | 7.634 | 6.514 | 11.614 | 3.013 |

## 2026-06-23 12:13:26 UTC | 0kAcEdn-C1M_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0kAcEdn-C1M_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `149.537` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.744 |
| save_clips | - |
| sample_frames | 0.820 |
| caption_frames | 32.858 |
| sample_fps | 2.060 |
| detect_object_yolo | 7.564 |
| audio_scan | 13.587 |
| asr_timings | 8.705 |
| ast_timings | 23.792 |
| describe_scenes | 24.579 |
| summarize_scenes | 10.073 |
| synthesize_synopsis | 20.350 |
| make_embedding | 3.044 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 33.684 |
| branch_yolo_total | 9.629 |
| branch_audio_total | 46.092 |

## 2026-06-27 14:06:22 UTC | 0kAcEdn-C1M_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0kAcEdn-C1M_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `122.639` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.779 |
| save_clips | - |
| sample_frames | 0.855 |
| caption_frames | 33.925 |
| sample_fps | 2.169 |
| detect_object_yolo | 7.967 |
| audio_scan | 14.105 |
| asr_timings | 8.091 |
| ast_timings | 24.498 |
| describe_scenes | 7.634 |
| summarize_scenes | 6.514 |
| synthesize_synopsis | 11.614 |
| make_embedding | 3.013 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 34.786 |
| branch_yolo_total | 10.141 |
| branch_audio_total | 46.702 |
