# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 17:24:26 UTC | 4jNdZg348kM_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 187.558 | 0.766 | 67.662 | 25.734 | 8.747 | 21.472 | 4.228 |
| 2026-06-24 11:17:02 UTC | 4jNdZg348kM_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 180.247 | 0.789 | 73.951 | 14.567 | 11.381 | 15.144 | 4.131 |

## 2026-06-23 17:24:26 UTC | 4jNdZg348kM_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/4jNdZg348kM_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `187.558` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.766 |
| save_clips | - |
| sample_frames | 1.398 |
| caption_frames | 44.364 |
| sample_fps | 2.372 |
| detect_object_yolo | 9.428 |
| audio_scan | 11.694 |
| asr_timings | 20.453 |
| ast_timings | 35.506 |
| describe_scenes | 25.734 |
| summarize_scenes | 8.747 |
| synthesize_synopsis | 21.472 |
| make_embedding | 4.228 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.768 |
| branch_yolo_total | 11.806 |
| branch_audio_total | 67.662 |

## 2026-06-24 11:17:02 UTC | 4jNdZg348kM_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/4jNdZg348kM_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `180.247` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.789 |
| save_clips | - |
| sample_frames | 1.414 |
| caption_frames | 45.559 |
| sample_fps | 2.387 |
| detect_object_yolo | 9.510 |
| audio_scan | 11.769 |
| asr_timings | 26.565 |
| ast_timings | 35.609 |
| describe_scenes | 14.567 |
| summarize_scenes | 11.381 |
| synthesize_synopsis | 15.144 |
| make_embedding | 4.131 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.978 |
| branch_yolo_total | 11.902 |
| branch_audio_total | 73.951 |
