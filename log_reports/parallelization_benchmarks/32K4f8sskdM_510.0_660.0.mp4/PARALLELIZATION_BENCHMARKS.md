# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 15:33:22 UTC | 32K4f8sskdM_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 262.922 | 0.624 | 95.858 | 37.073 | 22.762 | 28.349 | 5.484 |
| 2026-06-24 09:30:46 UTC | 32K4f8sskdM_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 246.311 | 0.621 | 96.124 | 29.718 | 23.991 | 15.781 | 5.452 |

## 2026-06-23 15:33:22 UTC | 32K4f8sskdM_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/32K4f8sskdM_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `262.922` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.624 |
| save_clips | - |
| sample_frames | 1.835 |
| caption_frames | 56.064 |
| sample_fps | 2.428 |
| detect_object_yolo | 11.067 |
| audio_scan | 13.692 |
| asr_timings | 39.403 |
| ast_timings | 42.754 |
| describe_scenes | 37.073 |
| summarize_scenes | 22.762 |
| synthesize_synopsis | 28.349 |
| make_embedding | 5.484 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 57.905 |
| branch_yolo_total | 13.500 |
| branch_audio_total | 95.858 |

## 2026-06-24 09:30:46 UTC | 32K4f8sskdM_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/32K4f8sskdM_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `246.311` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.621 |
| save_clips | - |
| sample_frames | 1.823 |
| caption_frames | 57.785 |
| sample_fps | 2.453 |
| detect_object_yolo | 11.136 |
| audio_scan | 13.866 |
| asr_timings | 38.969 |
| ast_timings | 43.281 |
| describe_scenes | 29.718 |
| summarize_scenes | 23.991 |
| synthesize_synopsis | 15.781 |
| make_embedding | 5.452 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 59.614 |
| branch_yolo_total | 13.594 |
| branch_audio_total | 96.124 |
