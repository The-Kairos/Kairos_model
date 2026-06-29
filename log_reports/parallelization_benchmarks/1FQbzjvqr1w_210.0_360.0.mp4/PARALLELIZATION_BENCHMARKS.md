# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 12:59:37 UTC | 1FQbzjvqr1w_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 125.649 | 0.806 | 58.630 | 9.814 | 8.173 | 21.631 | 1.781 |
| 2026-06-27 14:42:39 UTC | 1FQbzjvqr1w_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 105.620 | 0.803 | 61.593 | 4.344 | 4.131 | 6.511 | 1.847 |

## 2026-06-23 12:59:37 UTC | 1FQbzjvqr1w_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1FQbzjvqr1w_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `125.649` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.806 |
| save_clips | - |
| sample_frames | 0.501 |
| caption_frames | 15.118 |
| sample_fps | 1.920 |
| detect_object_yolo | 5.919 |
| audio_scan | 14.748 |
| asr_timings | 33.822 |
| ast_timings | 10.051 |
| describe_scenes | 9.814 |
| summarize_scenes | 8.173 |
| synthesize_synopsis | 21.631 |
| make_embedding | 1.781 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 15.625 |
| branch_yolo_total | 7.846 |
| branch_audio_total | 58.630 |

## 2026-06-27 14:42:39 UTC | 1FQbzjvqr1w_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1FQbzjvqr1w_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `105.620` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.803 |
| save_clips | - |
| sample_frames | 0.500 |
| caption_frames | 16.228 |
| sample_fps | 1.965 |
| detect_object_yolo | 6.216 |
| audio_scan | 15.179 |
| asr_timings | 35.890 |
| ast_timings | 10.515 |
| describe_scenes | 4.344 |
| summarize_scenes | 4.131 |
| synthesize_synopsis | 6.511 |
| make_embedding | 1.847 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 16.734 |
| branch_yolo_total | 8.188 |
| branch_audio_total | 61.593 |
