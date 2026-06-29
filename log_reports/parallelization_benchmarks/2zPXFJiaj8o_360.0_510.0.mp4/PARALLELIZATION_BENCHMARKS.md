# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 15:00:56 UTC | 2zPXFJiaj8o_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 110.745 | 0.646 | 35.286 | 7.768 | 11.869 | 23.217 | 2.158 |
| 2026-06-24 09:01:34 UTC | 2zPXFJiaj8o_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 117.585 | 0.657 | 34.979 | 7.652 | 11.475 | 30.579 | 2.098 |

## 2026-06-23 15:00:56 UTC | 2zPXFJiaj8o_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2zPXFJiaj8o_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `110.745` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.646 |
| save_clips | - |
| sample_frames | 0.408 |
| caption_frames | 19.372 |
| sample_fps | 1.748 |
| detect_object_yolo | 6.910 |
| audio_scan | 10.600 |
| asr_timings | 11.847 |
| ast_timings | 12.830 |
| describe_scenes | 7.768 |
| summarize_scenes | 11.869 |
| synthesize_synopsis | 23.217 |
| make_embedding | 2.158 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 19.786 |
| branch_yolo_total | 8.664 |
| branch_audio_total | 35.286 |

## 2026-06-24 09:01:34 UTC | 2zPXFJiaj8o_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2zPXFJiaj8o_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `117.585` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.657 |
| save_clips | - |
| sample_frames | 0.421 |
| caption_frames | 19.609 |
| sample_fps | 1.766 |
| detect_object_yolo | 6.974 |
| audio_scan | 10.691 |
| asr_timings | 11.485 |
| ast_timings | 12.795 |
| describe_scenes | 7.652 |
| summarize_scenes | 11.475 |
| synthesize_synopsis | 30.579 |
| make_embedding | 2.098 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 20.036 |
| branch_yolo_total | 8.745 |
| branch_audio_total | 34.979 |
