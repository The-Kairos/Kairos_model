# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-04-08 11:39:50 UTC | NEW_YORK_TIMES_SQUARE_2024___4K_WALK_TOUR_MORNING.mp4 | parallel | gemini | gemini-embedding-001 | 180.884 | 2.935 | 76.555 | 64.465 | 16.898 | 12.151 | 3.065 |

## 2026-04-08 11:39:50 UTC | NEW_YORK_TIMES_SQUARE_2024___4K_WALK_TOUR_MORNING.mp4 | parallel

- Video path: `/var/tmp/kairos/jobs/383be544-a3c2-4e67-951e-1768bafafac5/NEW_YORK_TIMES_SQUARE_2024___4K_WALK_TOUR_MORNING.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `180.884` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.935 |
| save_clips | - |
| sample_frames | 4.718 |
| caption_frames | 71.829 |
| sample_fps | 8.988 |
| detect_object_yolo | 45.327 |
| audio_scan | 31.971 |
| asr_timings | 29.886 |
| ast_timings | 28.970 |
| describe_scenes | 64.465 |
| summarize_scenes | 16.898 |
| synthesize_synopsis | 12.151 |
| make_embedding | 3.065 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 76.555 |
| branch_yolo_total | 54.322 |
| branch_audio_total | 61.870 |
