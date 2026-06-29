# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 13:03:09 UTC | 1FQbzjvqr1w_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 211.121 | 0.816 | 101.645 | 17.886 | 23.414 | 23.119 | 2.797 |
| 2026-06-27 14:44:46 UTC | 1FQbzjvqr1w_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 126.388 | 0.853 | 51.415 | 7.715 | 12.669 | 7.249 | 2.801 |

## 2026-06-23 13:03:09 UTC | 1FQbzjvqr1w_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1FQbzjvqr1w_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `211.121` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.816 |
| save_clips | - |
| sample_frames | 1.130 |
| caption_frames | 29.639 |
| sample_fps | 2.199 |
| detect_object_yolo | 7.108 |
| audio_scan | 13.697 |
| asr_timings | 66.808 |
| ast_timings | 21.128 |
| describe_scenes | 17.886 |
| summarize_scenes | 23.414 |
| synthesize_synopsis | 23.119 |
| make_embedding | 2.797 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.775 |
| branch_yolo_total | 9.313 |
| branch_audio_total | 101.645 |

## 2026-06-27 14:44:46 UTC | 1FQbzjvqr1w_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1FQbzjvqr1w_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `126.388` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.853 |
| save_clips | - |
| sample_frames | 1.170 |
| caption_frames | 31.165 |
| sample_fps | 2.268 |
| detect_object_yolo | 7.573 |
| audio_scan | 14.140 |
| asr_timings | 15.931 |
| ast_timings | 21.334 |
| describe_scenes | 7.715 |
| summarize_scenes | 12.669 |
| synthesize_synopsis | 7.249 |
| make_embedding | 2.801 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.341 |
| branch_yolo_total | 9.847 |
| branch_audio_total | 51.415 |
