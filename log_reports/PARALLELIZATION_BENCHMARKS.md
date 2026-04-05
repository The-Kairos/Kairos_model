# Parallelization Benchmarks

## 2026-04-05 15:18:55 UTC | Argentina v France Full Penalty Shoot-out.mp4 | semi_parallel

- Video path: `Videos/Argentina v France Full Penalty Shoot-out.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `317.158` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.572 |
| save_clips | 9.274 |
| sample_frames | 5.837 |
| caption_frames | 80.530 |
| sample_fps | 42.216 |
| detect_object_yolo | 26.660 |
| audio_scan | 41.670 |
| asr_timings | 21.388 |
| ast_timings | 18.714 |
| describe_scenes | 48.100 |
| summarize_scenes | 10.387 |
| synthesize_synopsis | 5.456 |
| make_embedding | 4.164 |
| 2026-04-05 15:27:12 UTC | Argentina v France Full Penalty Shoot-out.mp4 | parallel | gemini | gemini-embedding-001 | 216.328 | 52.131 | 9.198 | 6.101 | 4.195 |


## 2026-04-05 15:43:02 UTC | Argentina v France Full Penalty Shoot-out.mp4 | parallel

- Video path: `Videos/Argentina v France Full Penalty Shoot-out.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `221.984` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.553 |
| save_clips | 9.280 |
| sample_frames | 6.952 |
| caption_frames | 134.802 |
| sample_fps | 62.664 |
| detect_object_yolo | 32.119 |
| audio_scan | 25.877 |
| asr_timings | 31.838 |
| ast_timings | 15.532 |
| describe_scenes | 50.215 |
| summarize_scenes | 8.358 |
| synthesize_synopsis | 5.453 |
| make_embedding | 4.202 |
