# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-04-08 03:56:43 UTC | Young_Sheldon_-_First_Day_of_High_School.mp4 | semi_parallel | gemini | gemini-embedding-001 | 190.516 | 1.517 | 116.988 | 39.218 | 14.310 | 11.004 | 2.502 |
| 2026-04-08 04:04:02 UTC | Young_Sheldon_-_First_Day_of_High_School.mp4 | semi_parallel | gemini | gemini-embedding-001 | 191.168 | 1.494 | 118.016 | 40.233 | 16.779 | 7.142 | 2.487 |
| 2026-04-08 04:09:12 UTC | Young_Sheldon_-_First_Day_of_High_School.mp4 | semi_parallel | gemini | gemini-embedding-001 | 187.057 | 1.473 | 116.917 | 41.906 | 12.137 | 6.873 | 2.549 |
| 2026-04-08 04:20:16 UTC | Young_Sheldon_-_First_Day_of_High_School.mp4 | semi_parallel | gemini | gemini-embedding-001 | 193.660 | 1.529 | 116.689 | 34.532 | 23.166 | 10.135 | 2.515 |
| 2026-04-08 04:40:30 UTC | Young_Sheldon_-_First_Day_of_High_School.mp4 | semi_parallel | gemini | gemini-embedding-001 | 180.619 | 1.554 | 117.061 | 37.005 | 10.371 | 7.157 | 2.480 |
| 2026-04-08 05:14:58 UTC | Young_Sheldon_-_First_Day_of_High_School.mp4 | semi_parallel | gemini | gemini-embedding-001 | 187.609 | 1.471 | 116.970 | 41.148 | 10.670 | 9.762 | 2.561 |
| 2026-04-08 05:44:26 UTC | Young_Sheldon_-_First_Day_of_High_School.mp4 | semi_parallel | gemini | gemini-embedding-001 | 187.621 | 1.628 | 119.871 | 36.483 | 11.784 | 10.382 | 2.465 |
| 2026-04-08 05:50:37 UTC | Young_Sheldon_-_First_Day_of_High_School.mp4 | semi_parallel | gemini | gemini-embedding-001 | 208.352 | 1.501 | 118.993 | 57.729 | 10.244 | 12.696 | 2.382 |
| 2026-04-08 09:10:26 UTC | Young_Sheldon_-_First_Day_of_High_School.mp4 | parallel | gemini | gemini-embedding-001 | 129.480 | 1.495 | 42.905 | 48.665 | 16.431 | 10.627 | 4.726 |
| 2026-04-08 09:15:55 UTC | Young_Sheldon_-_First_Day_of_High_School.mp4 | parallel | gemini | gemini-embedding-001 | 139.630 | 1.488 | 42.948 | 46.422 | 18.518 | 23.273 | 2.428 |

## 2026-04-08 03:56:43 UTC | Young_Sheldon_-_First_Day_of_High_School.mp4 | semi_parallel

- Video path: `/var/tmp/kairos/jobs/c333dc89-f39f-4c37-8380-8bdce38fcc1d/Young_Sheldon_-_First_Day_of_High_School.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `190.516` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.517 |
| save_clips | - |
| sample_frames | 3.436 |
| caption_frames | 23.924 |
| sample_fps | 6.385 |
| detect_object_yolo | 11.852 |
| audio_scan | 14.244 |
| asr_timings | 10.090 |
| ast_timings | 47.040 |
| describe_scenes | 39.218 |
| summarize_scenes | 14.310 |
| synthesize_synopsis | 11.004 |
| make_embedding | 2.502 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 27.364 |
| branch_yolo_total | 18.242 |
| branch_audio_total | 71.381 |

## 2026-04-08 04:04:02 UTC | Young_Sheldon_-_First_Day_of_High_School.mp4 | semi_parallel

- Video path: `/var/tmp/kairos/jobs/2fd44a83-9775-4a35-b11c-2af2b8f64831/Young_Sheldon_-_First_Day_of_High_School.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `191.168` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.494 |
| save_clips | - |
| sample_frames | 3.376 |
| caption_frames | 23.574 |
| sample_fps | 6.291 |
| detect_object_yolo | 11.788 |
| audio_scan | 14.195 |
| asr_timings | 10.020 |
| ast_timings | 48.755 |
| describe_scenes | 40.233 |
| summarize_scenes | 16.779 |
| synthesize_synopsis | 7.142 |
| make_embedding | 2.487 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 26.954 |
| branch_yolo_total | 18.084 |
| branch_audio_total | 72.978 |

## 2026-04-08 04:09:12 UTC | Young_Sheldon_-_First_Day_of_High_School.mp4 | semi_parallel

- Video path: `/var/tmp/kairos/jobs/d674b507-4094-44cc-a686-908cc7763482/Young_Sheldon_-_First_Day_of_High_School.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `187.057` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.473 |
| save_clips | - |
| sample_frames | 3.385 |
| caption_frames | 22.999 |
| sample_fps | 6.389 |
| detect_object_yolo | 11.894 |
| audio_scan | 14.289 |
| asr_timings | 9.955 |
| ast_timings | 47.988 |
| describe_scenes | 41.906 |
| summarize_scenes | 12.137 |
| synthesize_synopsis | 6.873 |
| make_embedding | 2.549 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 26.389 |
| branch_yolo_total | 18.288 |
| branch_audio_total | 72.239 |

## 2026-04-08 04:20:16 UTC | Young_Sheldon_-_First_Day_of_High_School.mp4 | semi_parallel

- Video path: `/var/tmp/kairos/jobs/16b28df1-5d42-4ff4-9bda-89b862677a4e/Young_Sheldon_-_First_Day_of_High_School.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `193.660` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.529 |
| save_clips | - |
| sample_frames | 3.429 |
| caption_frames | 22.851 |
| sample_fps | 6.278 |
| detect_object_yolo | 11.791 |
| audio_scan | 14.210 |
| asr_timings | 10.843 |
| ast_timings | 47.271 |
| describe_scenes | 34.532 |
| summarize_scenes | 23.166 |
| synthesize_synopsis | 10.135 |
| make_embedding | 2.515 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 26.285 |
| branch_yolo_total | 18.073 |
| branch_audio_total | 72.331 |

## 2026-04-08 04:40:30 UTC | Young_Sheldon_-_First_Day_of_High_School.mp4 | semi_parallel

- Video path: `/var/tmp/kairos/jobs/a21c0e1a-5c5a-4f28-a9fc-b501203a930a/Young_Sheldon_-_First_Day_of_High_School.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `180.619` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.554 |
| save_clips | - |
| sample_frames | 3.406 |
| caption_frames | 22.576 |
| sample_fps | 6.337 |
| detect_object_yolo | 11.866 |
| audio_scan | 14.237 |
| asr_timings | 10.595 |
| ast_timings | 48.027 |
| describe_scenes | 37.005 |
| summarize_scenes | 10.371 |
| synthesize_synopsis | 7.157 |
| make_embedding | 2.480 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 25.987 |
| branch_yolo_total | 18.208 |
| branch_audio_total | 72.866 |

## 2026-04-08 05:14:58 UTC | Young_Sheldon_-_First_Day_of_High_School.mp4 | semi_parallel

- Video path: `/var/tmp/kairos/jobs/ca9ce1a1-5d3c-4359-bf56-895e4d3501da/Young_Sheldon_-_First_Day_of_High_School.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `187.609` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.471 |
| save_clips | - |
| sample_frames | 3.439 |
| caption_frames | 22.947 |
| sample_fps | 6.231 |
| detect_object_yolo | 11.711 |
| audio_scan | 14.130 |
| asr_timings | 10.544 |
| ast_timings | 47.951 |
| describe_scenes | 41.148 |
| summarize_scenes | 10.670 |
| synthesize_synopsis | 9.762 |
| make_embedding | 2.561 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 26.391 |
| branch_yolo_total | 17.947 |
| branch_audio_total | 72.633 |

## 2026-04-08 05:44:26 UTC | Young_Sheldon_-_First_Day_of_High_School.mp4 | semi_parallel

- Video path: `/var/tmp/kairos/jobs/3ff232c0-6c2e-4a3a-bbe1-5fb097316eb9/Young_Sheldon_-_First_Day_of_High_School.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `187.621` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.628 |
| save_clips | - |
| sample_frames | 3.367 |
| caption_frames | 25.727 |
| sample_fps | 6.341 |
| detect_object_yolo | 11.584 |
| audio_scan | 14.443 |
| asr_timings | 10.125 |
| ast_timings | 48.266 |
| describe_scenes | 36.483 |
| summarize_scenes | 11.784 |
| synthesize_synopsis | 10.382 |
| make_embedding | 2.465 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 29.099 |
| branch_yolo_total | 17.930 |
| branch_audio_total | 72.841 |

## 2026-04-08 05:50:37 UTC | Young_Sheldon_-_First_Day_of_High_School.mp4 | semi_parallel

- Video path: `/var/tmp/kairos/jobs/9f348988-bb28-4211-8613-a95d66c2826c/Young_Sheldon_-_First_Day_of_High_School.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `208.352` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.501 |
| save_clips | - |
| sample_frames | 3.370 |
| caption_frames | 25.630 |
| sample_fps | 6.220 |
| detect_object_yolo | 12.584 |
| audio_scan | 12.858 |
| asr_timings | 9.921 |
| ast_timings | 48.392 |
| describe_scenes | 57.729 |
| summarize_scenes | 10.244 |
| synthesize_synopsis | 12.696 |
| make_embedding | 2.382 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 29.006 |
| branch_yolo_total | 18.809 |
| branch_audio_total | 71.178 |

## 2026-04-08 09:10:26 UTC | Young_Sheldon_-_First_Day_of_High_School.mp4 | parallel

- Video path: `/var/tmp/kairos/jobs/afc9bdf1-5564-45b6-9310-558bf050bffb/Young_Sheldon_-_First_Day_of_High_School.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `129.480` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.495 |
| save_clips | - |
| sample_frames | 4.206 |
| caption_frames | 38.691 |
| sample_fps | 8.386 |
| detect_object_yolo | 17.293 |
| audio_scan | 22.340 |
| asr_timings | 13.380 |
| ast_timings | 17.060 |
| describe_scenes | 48.665 |
| summarize_scenes | 16.431 |
| synthesize_synopsis | 10.627 |
| make_embedding | 4.726 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 42.905 |
| branch_yolo_total | 25.687 |
| branch_audio_total | 39.410 |

## 2026-04-08 09:15:55 UTC | Young_Sheldon_-_First_Day_of_High_School.mp4 | parallel

- Video path: `/var/tmp/kairos/jobs/ac25b030-731f-4437-867b-57d2d70b5ead/Young_Sheldon_-_First_Day_of_High_School.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `139.630` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.488 |
| save_clips | - |
| sample_frames | 4.158 |
| caption_frames | 38.782 |
| sample_fps | 8.429 |
| detect_object_yolo | 17.124 |
| audio_scan | 21.809 |
| asr_timings | 13.645 |
| ast_timings | 16.917 |
| describe_scenes | 46.422 |
| summarize_scenes | 18.518 |
| synthesize_synopsis | 23.273 |
| make_embedding | 2.428 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 42.948 |
| branch_yolo_total | 25.560 |
| branch_audio_total | 38.741 |
