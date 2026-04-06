# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2026-04-05 15:18:55 UTC | Argentina v France Full Penalty Shoot-out.mp4 | semi_parallel | gemini | gemini-embedding-001 | 317.158 | 48.100 | 10.387 | 5.456 | 4.164 |
| 2026-04-05 15:43:02 UTC | Argentina v France Full Penalty Shoot-out.mp4 | parallel | gemini | gemini-embedding-001 | 221.984 | 50.215 | 8.358 | 5.453 | 4.202 |
| 2026-04-05 17:08:34 UTC | Argentina v France Full Penalty Shoot-out.mp4 | semi_parallel | gemini | gemini-embedding-001 | 280.410 | 50.782 | 10.246 | 5.630 | 4.033 |
| 2026-04-05 17:13:49 UTC | Argentina v France Full Penalty Shoot-out.mp4 | parallel | gemini | gemini-embedding-001 | 213.151 | 48.156 | 9.578 | 7.479 | 3.903 |
| 2026-04-06 06:12:38 UTC | Young Sheldon - First Day of High School.mp4 | semi_parallel | gemini | gemini-embedding-001 | 167.580 | 33.735 | 8.759 | 8.124 | 2.843 |
| 2026-04-06 06:16:38 UTC | Young Sheldon - First Day of High School.mp4 | parallel | gemini | gemini-embedding-001 | 132.463 | 32.277 | 9.312 | 9.143 | 2.804 |

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

## ---- After stopping debug artifacts from being saved ----

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

## 2026-04-06 06:12:38 UTC | Young Sheldon - First Day of High School.mp4 | semi_parallel

- Video path: `Videos/Young Sheldon - First Day of High School.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `167.580` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.490 |
| save_clips | - |
| sample_frames | 3.353 |
| caption_frames | 39.741 |
| sample_fps | 18.442 |
| detect_object_yolo | 11.542 |
| audio_scan | 21.360 |
| asr_timings | 10.001 |
| ast_timings | 8.009 |
| describe_scenes | 33.735 |
| summarize_scenes | 8.759 |
| synthesize_synopsis | 8.124 |
| make_embedding | 2.843 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.101 |
| branch_yolo_total | 29.989 |
| branch_audio_total | 39.378 |

## 2026-04-06 06:16:38 UTC | Young Sheldon - First Day of High School.mp4 | parallel

- Video path: `Videos/Young Sheldon - First Day of High School.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `132.463` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.465 |
| save_clips | - |
| sample_frames | 4.695 |
| caption_frames | 72.596 |
| sample_fps | 27.873 |
| detect_object_yolo | 14.291 |
| audio_scan | 15.871 |
| asr_timings | 15.960 |
| ast_timings | 8.850 |
| describe_scenes | 32.277 |
| summarize_scenes | 9.312 |
| synthesize_synopsis | 9.143 |
| make_embedding | 2.804 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 77.302 |
| branch_yolo_total | 42.173 |
| branch_audio_total | 31.841 |
