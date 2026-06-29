# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-04-10 12:24:04 UTC | NEW_YORK_TIMES_SQUARE_2024___4K_WALK_TOUR_MORNING_clip.mp4 | parallel | gemini | gemini-embedding-001 | 37.202 | 0.084 | 3.191 | 10.054 | 4.765 | 14.063 | 0.689 |

## 2026-04-10 12:24:04 UTC | NEW_YORK_TIMES_SQUARE_2024___4K_WALK_TOUR_MORNING_clip.mp4 | parallel

- Video path: `/home/Kairos_model/.tmp/kairos/jobs/063016fb-9e64-407d-b4b2-3c8633e366d8/NEW_YORK_TIMES_SQUARE_2024___4K_WALK_TOUR_MORNING_clip.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `37.202` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.084 |
| save_clips | - |
| sample_frames | 0.045 |
| caption_frames | 1.300 |
| sample_fps | 0.070 |
| detect_object_yolo | 0.940 |
| audio_scan | 0.120 |
| asr_timings | 0.000 |
| ast_timings | 3.061 |
| describe_scenes | 10.054 |
| summarize_scenes | 4.765 |
| synthesize_synopsis | 14.063 |
| make_embedding | 0.689 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 1.352 |
| branch_yolo_total | 1.018 |
| branch_audio_total | 3.191 |
