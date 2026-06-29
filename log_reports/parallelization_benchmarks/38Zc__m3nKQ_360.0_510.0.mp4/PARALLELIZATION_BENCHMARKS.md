# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 15:40:58 UTC | 38Zc__m3nKQ_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 115.577 | 0.677 | 34.468 | 11.652 | 14.476 | 27.881 | 1.830 |
| 2026-06-24 09:37:39 UTC | 38Zc__m3nKQ_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 109.057 | 0.674 | 33.867 | 7.518 | 9.499 | 30.380 | 1.805 |

## 2026-06-23 15:40:58 UTC | 38Zc__m3nKQ_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/38Zc__m3nKQ_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `115.577` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.677 |
| save_clips | - |
| sample_frames | 0.357 |
| caption_frames | 15.071 |
| sample_fps | 1.742 |
| detect_object_yolo | 6.043 |
| audio_scan | 14.669 |
| asr_timings | 10.330 |
| ast_timings | 9.461 |
| describe_scenes | 11.652 |
| summarize_scenes | 14.476 |
| synthesize_synopsis | 27.881 |
| make_embedding | 1.830 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 15.433 |
| branch_yolo_total | 7.790 |
| branch_audio_total | 34.468 |

## 2026-06-24 09:37:39 UTC | 38Zc__m3nKQ_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/38Zc__m3nKQ_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `109.057` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.674 |
| save_clips | - |
| sample_frames | 0.354 |
| caption_frames | 15.633 |
| sample_fps | 1.754 |
| detect_object_yolo | 6.159 |
| audio_scan | 14.938 |
| asr_timings | 9.459 |
| ast_timings | 9.462 |
| describe_scenes | 7.518 |
| summarize_scenes | 9.499 |
| synthesize_synopsis | 30.380 |
| make_embedding | 1.805 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 15.993 |
| branch_yolo_total | 7.919 |
| branch_audio_total | 33.867 |
