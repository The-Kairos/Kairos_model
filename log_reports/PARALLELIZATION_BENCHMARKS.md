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

## After stopping debug srtifacts from being saved:

| 2026-04-05 17:08:34 UTC | Argentina v France Full Penalty Shoot-out.mp4 | semi_parallel | gemini | gemini-embedding-001 | 280.410 | 50.782 | 10.246 | 5.630 | 4.033 |

## 2026-04-05 17:08:34 UTC | Argentina v France Full Penalty Shoot-out.mp4 | semi_parallel

- Video path: `Videos/Argentina v France Full Penalty Shoot-out.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `280.410` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.487 |
| save_clips | - |
| sample_frames | 5.732 |
| caption_frames | 76.278 |
| sample_fps | 41.852 |
| detect_object_yolo | 25.719 |
| audio_scan | 22.256 |
| asr_timings | 20.532 |
| ast_timings | 14.677 |
| describe_scenes | 50.782 |
| summarize_scenes | 10.246 |
| synthesize_synopsis | 5.630 |
| make_embedding | 4.033 |
| 2026-04-05 17:13:49 UTC | Argentina v France Full Penalty Shoot-out.mp4 | parallel | gemini | gemini-embedding-001 | 213.151 | 48.156 | 9.578 | 7.479 | 3.903 |

## 2026-04-05 17:13:49 UTC | Argentina v France Full Penalty Shoot-out.mp4 | parallel

- Video path: `Videos/Argentina v France Full Penalty Shoot-out.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `213.151` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.484 |
| save_clips | - |
| sample_frames | 6.780 |
| caption_frames | 134.605 |
| sample_fps | 59.876 |
| detect_object_yolo | 30.803 |
| audio_scan | 25.501 |
| asr_timings | 32.383 |
| ast_timings | 15.633 |
| describe_scenes | 48.156 |
| summarize_scenes | 9.578 |
| synthesize_synopsis | 7.479 |
| make_embedding | 3.903 |
