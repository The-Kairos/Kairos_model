# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 14:10:53 UTC | 1m3-4Bi5Kl0_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 154.764 | 0.773 | 44.656 | 28.159 | 11.902 | 26.961 | 2.823 |
| 2026-06-27 15:31:56 UTC | 1m3-4Bi5Kl0_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 113.034 | 0.809 | 45.339 | 7.243 | 6.766 | 9.105 | 2.753 |

## 2026-06-23 14:10:53 UTC | 1m3-4Bi5Kl0_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1m3-4Bi5Kl0_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `154.764` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.773 |
| save_clips | - |
| sample_frames | 0.766 |
| caption_frames | 27.841 |
| sample_fps | 2.078 |
| detect_object_yolo | 7.437 |
| audio_scan | 14.799 |
| asr_timings | 8.550 |
| ast_timings | 21.299 |
| describe_scenes | 28.159 |
| summarize_scenes | 11.902 |
| synthesize_synopsis | 26.961 |
| make_embedding | 2.823 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 28.613 |
| branch_yolo_total | 9.521 |
| branch_audio_total | 44.656 |

## 2026-06-27 15:31:56 UTC | 1m3-4Bi5Kl0_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1m3-4Bi5Kl0_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `113.034` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.809 |
| save_clips | - |
| sample_frames | 0.782 |
| caption_frames | 29.108 |
| sample_fps | 2.110 |
| detect_object_yolo | 7.619 |
| audio_scan | 15.000 |
| asr_timings | 8.604 |
| ast_timings | 21.727 |
| describe_scenes | 7.243 |
| summarize_scenes | 6.766 |
| synthesize_synopsis | 9.105 |
| make_embedding | 2.753 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 29.896 |
| branch_yolo_total | 9.735 |
| branch_audio_total | 45.339 |
