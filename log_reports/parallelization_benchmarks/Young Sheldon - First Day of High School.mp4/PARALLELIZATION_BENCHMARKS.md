# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-04-06 07:05:09 UTC | Young Sheldon - First Day of High School.mp4 | semi_parallel | gemini | gemini-embedding-001 | 161.565 | 1.514 | 41.046 | 34.601 | 10.179 | 9.483 | 2.515 |
| 2026-04-06 07:07:31 UTC | Young Sheldon - First Day of High School.mp4 | parallel | gemini | gemini-embedding-001 | 128.822 | 1.469 | 71.880 | 34.820 | 9.432 | 8.521 | 2.535 |
| 2026-04-06 11:16:08 UTC | Young Sheldon - First Day of High School.mp4 | parallel | gemini | gemini-embedding-001 | 141.628 | 1.464 | 73.563 | 43.704 | 12.170 | 7.802 | 2.539 |
| 2026-04-06 14:26:11 UTC | Young Sheldon - First Day of High School.mp4 | parallel | gemini | gemini-embedding-001 | 145.065 | 1.487 | 72.367 | 42.287 | 13.241 | 12.641 | 2.659 |
| 2026-04-06 14:33:11 UTC | Young Sheldon - First Day of High School.mp4 | parallel | gemini | gemini-embedding-001 | 117.036 | 1.493 | 46.715 | 43.315 | 12.710 | 9.825 | 2.589 |
| 2026-04-06 14:39:06 UTC | Young Sheldon - First Day of High School.mp4 | parallel | gemini | gemini-embedding-001 | 122.444 | 1.479 | 47.831 | 46.381 | 12.142 | 11.629 | 2.584 |
| 2026-04-06 15:34:52 UTC | Young Sheldon - First Day of High School.mp4 | parallel | gemini | gemini-embedding-001 | 151.239 | 1.479 | 49.238 | 62.395 | 20.898 | 14.061 | 2.780 |
| 2026-04-06 16:12:28 UTC | Young Sheldon - First Day of High School.mp4 | parallel | gemini | gemini-embedding-001 | 127.501 | 1.471 | 49.743 | 39.821 | 17.367 | 16.137 | 2.574 |
| 2026-04-06 16:47:02 UTC | Young Sheldon - First Day of High School.mp4 | parallel | gemini | gemini-embedding-001 | 175.878 | 1.494 | 49.069 | 92.752 | 14.707 | 14.846 | 2.617 |
| 2026-04-06 16:54:59 UTC | Young Sheldon - First Day of High School.mp4 | parallel | gemini | gemini-embedding-001 | 138.989 | 1.527 | 49.047 | 47.822 | 27.231 | 10.431 | 2.540 |
| 2026-04-06 17:01:27 UTC | Young Sheldon - First Day of High School.mp4 | parallel | gemini | gemini-embedding-001 | 130.720 | 1.500 | 48.952 | 53.262 | 12.821 | 11.138 | 2.654 |
| 2026-04-06 19:05:42 UTC | Young Sheldon - First Day of High School.mp4 | parallel | gemini | gemini-embedding-001 | 125.205 | 1.503 | 48.721 | 43.613 | 14.302 | 14.162 | 2.516 |



## 2026-04-06 07:05:09 UTC | Young Sheldon - First Day of High School.mp4 | semi_parallel

- Video path: `Videos/Young Sheldon - First Day of High School.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `161.565` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.514 |
| save_clips | - |
| sample_frames | 3.369 |
| caption_frames | 37.677 |
| sample_fps | 18.396 |
| detect_object_yolo | 11.559 |
| audio_scan | 14.268 |
| asr_timings | 10.030 |
| ast_timings | 7.798 |
| describe_scenes | 34.601 |
| summarize_scenes | 10.179 |
| synthesize_synopsis | 9.483 |
| make_embedding | 2.515 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.046 |
| branch_yolo_total | 29.955 |
| branch_audio_total | 32.096 |

## --- Initial Parallelization ---

## 2026-04-06 07:07:31 UTC | Young Sheldon - First Day of High School.mp4 | parallel

- Video path: `Videos/Young Sheldon - First Day of High School.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `128.822` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.469 |
| save_clips | - |
| sample_frames | 4.680 |
| caption_frames | 67.200 |
| sample_fps | 28.293 |
| detect_object_yolo | 14.580 |
| audio_scan | 16.128 |
| asr_timings | 15.896 |
| ast_timings | 8.658 |
| describe_scenes | 34.820 |
| summarize_scenes | 9.432 |
| synthesize_synopsis | 8.521 |
| make_embedding | 2.535 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 71.880 |
| branch_yolo_total | 42.873 |
| branch_audio_total | 32.024 |

## --- GPU-enabled Parallelization

## 2026-04-06 11:16:08 UTC | Young Sheldon - First Day of High School.mp4 | parallel

- Video path: `Videos/Young Sheldon - First Day of High School.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `141.628` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.464 |
| save_clips | - |
| sample_frames | 4.521 |
| caption_frames | 69.036 |
| sample_fps | 28.519 |
| detect_object_yolo | 14.536 |
| audio_scan | 16.464 |
| asr_timings | 16.023 |
| ast_timings | 8.893 |
| describe_scenes | 43.704 |
| summarize_scenes | 12.170 |
| synthesize_synopsis | 7.802 |
| make_embedding | 2.539 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 73.563 |
| branch_yolo_total | 43.063 |
| branch_audio_total | 32.496 |

## --- Batched BLIP Processing (batches = [1,4,8])---

## 2026-04-06 14:26:11 UTC | Young Sheldon - First Day of High School.mp4 | parallel

- Video path: `Videos/Young Sheldon - First Day of High School.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `145.065` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.487 |
| save_clips | - |
| sample_frames | 4.534 |
| caption_frames | 67.827 |
| sample_fps | 28.525 |
| detect_object_yolo | 14.489 |
| audio_scan | 16.665 |
| asr_timings | 14.995 |
| ast_timings | 8.784 |
| describe_scenes | 42.287 |
| summarize_scenes | 13.241 |
| synthesize_synopsis | 12.641 |
| make_embedding | 2.659 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 72.367 |
| branch_yolo_total | 43.021 |
| branch_audio_total | 31.669 |

## 2026-04-06 14:33:11 UTC | Young Sheldon - First Day of High School.mp4 | parallel

- Video path: `Videos/Young Sheldon - First Day of High School.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `117.036` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.493 |
| save_clips | - |
| sample_frames | 4.572 |
| caption_frames | 38.486 |
| sample_fps | 31.711 |
| detect_object_yolo | 14.997 |
| audio_scan | 18.641 |
| asr_timings | 16.443 |
| ast_timings | 8.539 |
| describe_scenes | 43.315 |
| summarize_scenes | 12.710 |
| synthesize_synopsis | 9.825 |
| make_embedding | 2.589 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.067 |
| branch_yolo_total | 46.715 |
| branch_audio_total | 35.094 |

## 2026-04-06 14:39:06 UTC | Young Sheldon - First Day of High School.mp4 | parallel

- Video path: `Videos/Young Sheldon - First Day of High School.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `122.444` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.479 |
| save_clips | - |
| sample_frames | 4.514 |
| caption_frames | 43.311 |
| sample_fps | 31.553 |
| detect_object_yolo | 15.300 |
| audio_scan | 18.297 |
| asr_timings | 17.092 |
| ast_timings | 8.557 |
| describe_scenes | 46.381 |
| summarize_scenes | 12.142 |
| synthesize_synopsis | 11.629 |
| make_embedding | 2.584 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.831 |
| branch_yolo_total | 46.860 |
| branch_audio_total | 35.400 |


## --- Implementing a Less Seek-heavy YOLO Frame Sampling Strategy (Sequential) ---

## 2026-04-06 15:34:52 UTC | Young Sheldon - First Day of High School.mp4 | parallel

- Video path: `Videos/Young Sheldon - First Day of High School.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `151.239` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.479 |
| save_clips | - |
| sample_frames | 4.311 |
| caption_frames | 44.920 |
| sample_fps | 7.810 |
| detect_object_yolo | 15.079 |
| audio_scan | 23.660 |
| asr_timings | 14.553 |
| ast_timings | 7.959 |
| describe_scenes | 62.395 |
| summarize_scenes | 20.898 |
| synthesize_synopsis | 14.061 |
| make_embedding | 2.780 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.238 |
| branch_yolo_total | 22.895 |
| branch_audio_total | 38.222 |

## 2026-04-06 16:12:28 UTC | Young Sheldon - First Day of High School.mp4 | parallel

- Video path: `Videos/Young Sheldon - First Day of High School.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `127.501` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.471 |
| save_clips | - |
| sample_frames | 4.235 |
| caption_frames | 45.502 |
| sample_fps | 7.829 |
| detect_object_yolo | 15.207 |
| audio_scan | 23.436 |
| asr_timings | 14.590 |
| ast_timings | 7.871 |
| describe_scenes | 39.821 |
| summarize_scenes | 17.367 |
| synthesize_synopsis | 16.137 |
| make_embedding | 2.574 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.743 |
| branch_yolo_total | 23.042 |
| branch_audio_total | 38.035 |

## --- After tuning scene description workers & description + LLM cooldown 

### --- workers=8, cooldown=5
## 2026-04-06 16:47:02 UTC | Young Sheldon - First Day of High School.mp4 | parallel

- Video path: `Videos/Young Sheldon - First Day of High School.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `175.878` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.494 |
| save_clips | - |
| sample_frames | 4.243 |
| caption_frames | 44.821 |
| sample_fps | 7.860 |
| detect_object_yolo | 15.313 |
| audio_scan | 23.445 |
| asr_timings | 14.858 |
| ast_timings | 7.953 |
| describe_scenes | 92.752 |
| summarize_scenes | 14.707 |
| synthesize_synopsis | 14.846 |
| make_embedding | 2.617 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.069 |
| branch_yolo_total | 23.180 |
| branch_audio_total | 38.312 |

### workers=8, cooldown=1

## 2026-04-06 16:54:59 UTC | Young Sheldon - First Day of High School.mp4 | parallel

- Video path: `Videos/Young Sheldon - First Day of High School.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `138.989` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.527 |
| save_clips | - |
| sample_frames | 4.292 |
| caption_frames | 44.749 |
| sample_fps | 7.778 |
| detect_object_yolo | 15.363 |
| audio_scan | 23.917 |
| asr_timings | 13.979 |
| ast_timings | 7.950 |
| describe_scenes | 47.822 |
| summarize_scenes | 27.231 |
| synthesize_synopsis | 10.431 |
| make_embedding | 2.540 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.047 |
| branch_yolo_total | 23.148 |
| branch_audio_total | 37.905 |


## workers=10, cooldown=1

## 2026-04-06 17:01:27 UTC | Young Sheldon - First Day of High School.mp4 | parallel

- Video path: `Videos/Young Sheldon - First Day of High School.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `130.720` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.500 |
| save_clips | - |
| sample_frames | 4.322 |
| caption_frames | 44.624 |
| sample_fps | 7.831 |
| detect_object_yolo | 15.185 |
| audio_scan | 23.565 |
| asr_timings | 14.506 |
| ast_timings | 7.875 |
| describe_scenes | 53.262 |
| summarize_scenes | 12.821 |
| synthesize_synopsis | 11.138 |
| make_embedding | 2.654 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 48.952 |
| branch_yolo_total | 23.022 |
| branch_audio_total | 38.080 |


### --- One more batch size 4 run just to test again

## 2026-04-06 18:48:45 UTC | Young Sheldon - First Day of High School.mp4 | parallel

## 2026-04-06 19:05:42 UTC | Young Sheldon - First Day of High School.mp4 | parallel

- Video path: `Videos/Young Sheldon - First Day of High School.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `125.205` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.503 |
| save_clips | - |
| sample_frames | 4.324 |
| caption_frames | 44.390 |
| sample_fps | 7.775 |
| detect_object_yolo | 14.975 |
| audio_scan | 23.617 |
| asr_timings | 14.230 |
| ast_timings | 7.868 |
| describe_scenes | 43.613 |
| summarize_scenes | 14.302 |
| synthesize_synopsis | 14.162 |
| make_embedding | 2.516 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 48.721 |
| branch_yolo_total | 22.757 |
| branch_audio_total | 37.856 |
