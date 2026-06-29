# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 16:38:28 UTC | 3waJEHGNYug_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 137.520 | 0.528 | 38.246 | 13.158 | 26.321 | 20.562 | 2.607 |
| 2026-06-24 10:33:40 UTC | 3waJEHGNYug_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 132.003 | 0.530 | 38.714 | 8.785 | 13.607 | 30.983 | 2.541 |

## 2026-06-23 16:38:28 UTC | 3waJEHGNYug_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3waJEHGNYug_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `137.520` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.528 |
| save_clips | - |
| sample_frames | 0.574 |
| caption_frames | 26.022 |
| sample_fps | 1.547 |
| detect_object_yolo | 6.503 |
| audio_scan | 13.116 |
| asr_timings | 6.389 |
| ast_timings | 18.732 |
| describe_scenes | 13.158 |
| summarize_scenes | 26.321 |
| synthesize_synopsis | 20.562 |
| make_embedding | 2.607 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 26.602 |
| branch_yolo_total | 8.056 |
| branch_audio_total | 38.246 |

## 2026-06-24 10:33:40 UTC | 3waJEHGNYug_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3waJEHGNYug_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `132.003` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.530 |
| save_clips | - |
| sample_frames | 0.571 |
| caption_frames | 26.627 |
| sample_fps | 1.582 |
| detect_object_yolo | 6.608 |
| audio_scan | 13.186 |
| asr_timings | 6.775 |
| ast_timings | 18.744 |
| describe_scenes | 8.785 |
| summarize_scenes | 13.607 |
| synthesize_synopsis | 30.983 |
| make_embedding | 2.541 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 27.203 |
| branch_yolo_total | 8.196 |
| branch_audio_total | 38.714 |
