# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-04-10 12:38:47 UTC | NEW_YORK_TIMES_SQUARE_2024___4K_WALK_TOUR_MORNING_clip__1_.mp4 | parallel | gemini | gemini-embedding-001 | 39.497 | 0.149 | 3.923 | 15.195 | 6.671 | 8.484 | 0.731 |

## 2026-04-10 12:38:47 UTC | NEW_YORK_TIMES_SQUARE_2024___4K_WALK_TOUR_MORNING_clip__1_.mp4 | parallel

- Video path: `/home/Kairos_model/.tmp/kairos/jobs/ccf255b4-d5c8-4ee0-a132-11af848c1adb/NEW_YORK_TIMES_SQUARE_2024___4K_WALK_TOUR_MORNING_clip__1_.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `39.497` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.149 |
| save_clips | - |
| sample_frames | 0.152 |
| caption_frames | 2.806 |
| sample_fps | 0.315 |
| detect_object_yolo | 2.174 |
| audio_scan | 0.766 |
| asr_timings | 0.000 |
| ast_timings | 3.148 |
| describe_scenes | 15.195 |
| summarize_scenes | 6.671 |
| synthesize_synopsis | 8.484 |
| make_embedding | 0.731 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 2.968 |
| branch_yolo_total | 2.496 |
| branch_audio_total | 3.923 |
