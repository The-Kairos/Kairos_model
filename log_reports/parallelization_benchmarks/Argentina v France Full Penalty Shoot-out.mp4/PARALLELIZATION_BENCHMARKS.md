# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-04-06 07:13:01 UTC | Argentina v France Full Penalty Shoot-out.mp4 | semi_parallel | gemini | gemini-embedding-001 | 317.074 | 2.497 | 84.340 | 79.458 | 13.840 | 8.087 | 3.853 |
| 2026-04-06 07:17:21 UTC | Argentina v France Full Penalty Shoot-out.mp4 | parallel | gemini | gemini-embedding-001 | 246.552 | 2.476 | 140.177 | 74.794 | 12.942 | 12.055 | 3.940 |
| 2026-04-06 11:20:23 UTC | Argentina v France Full Penalty Shoot-out.mp4 | parallel | gemini | gemini-embedding-001 | 241.338 | 2.548 | 144.350 | 65.616 | 13.217 | 11.118 | 4.088 |
| 2026-04-06 14:31:01 UTC | Argentina v France Full Penalty Shoot-out.mp4 | parallel | gemini | gemini-embedding-001 | 277.033 | 2.559 | 141.449 | 84.730 | 30.992 | 12.798 | 4.107 |
| 2026-04-06 14:36:51 UTC | Argentina v France Full Penalty Shoot-out.mp4 | parallel | gemini | gemini-embedding-001 | 206.972 | 2.505 | 96.794 | 78.893 | 14.636 | 9.454 | 4.294 |
| 2026-04-06 14:43:02 UTC | Argentina v France Full Penalty Shoot-out.mp4 | parallel | gemini | gemini-embedding-001 | 222.339 | 2.531 | 96.699 | 79.686 | 26.917 | 12.064 | 4.047 |
| 2026-04-06 15:39:04 UTC | Argentina v France Full Penalty Shoot-out.mp4 | parallel | gemini | gemini-embedding-001 | 238.962 | 2.563 | 99.372 | 95.045 | 21.786 | 15.618 | 4.177 |
| 2026-04-06 16:16:21 UTC | Argentina v France Full Penalty Shoot-out.mp4 | parallel | gemini | gemini-embedding-001 | 219.780 | 2.531 | 99.176 | 86.275 | 15.266 | 11.968 | 4.158 |
| 2026-04-06 16:52:28 UTC | Argentina v France Full Penalty Shoot-out.mp4 | parallel | gemini | gemini-embedding-001 | 312.253 | 2.510 | 99.872 | 176.435 | 15.714 | 13.126 | 4.187 |
| 2026-04-06 16:59:03 UTC | Argentina v France Full Penalty Shoot-out.mp4 | parallel | gemini | gemini-embedding-001 | 230.812 | 2.520 | 99.446 | 99.506 | 15.065 | 9.679 | 4.197 |
| 2026-04-06 17:05:20 UTC | Argentina v France Full Penalty Shoot-out.mp4 | parallel | gemini | gemini-embedding-001 | 219.944 | 2.491 | 99.776 | 81.495 | 19.445 | 12.225 | 4.103 |
| 2026-04-06 19:09:50 UTC | Argentina v France Full Penalty Shoot-out.mp4 | parallel | gemini | gemini-embedding-001 | 235.286 | 2.523 | 100.320 | 93.626 | 22.756 | 11.500 | 4.157 |

## 2026-04-06 07:13:01 UTC | Argentina v France Full Penalty Shoot-out.mp4 | semi_parallel

- Video path: `Videos/Argentina v France Full Penalty Shoot-out.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `317.074` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.497 |
| save_clips | - |
| sample_frames | 5.691 |
| caption_frames | 78.649 |
| sample_fps | 41.228 |
| detect_object_yolo | 25.924 |
| audio_scan | 22.505 |
| asr_timings | 20.562 |
| ast_timings | 14.591 |
| describe_scenes | 79.458 |
| summarize_scenes | 13.840 |
| synthesize_synopsis | 8.087 |
| make_embedding | 3.853 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 84.340 |
| branch_yolo_total | 67.152 |
| branch_audio_total | 57.658 |

## --- Initial Parallelization ---

## 2026-04-06 07:17:21 UTC | Argentina v France Full Penalty Shoot-out.mp4 | parallel

- Video path: `Videos/Argentina v France Full Penalty Shoot-out.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `246.552` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.476 |
| save_clips | - |
| sample_frames | 6.771 |
| caption_frames | 133.406 |
| sample_fps | 61.148 |
| detect_object_yolo | 30.717 |
| audio_scan | 26.072 |
| asr_timings | 32.908 |
| ast_timings | 14.956 |
| describe_scenes | 74.794 |
| summarize_scenes | 12.942 |
| synthesize_synopsis | 12.055 |
| make_embedding | 3.940 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 140.177 |
| branch_yolo_total | 91.865 |
| branch_audio_total | 58.980 |


## --- GPU-enabled Parallelization

## 2026-04-06 11:20:23 UTC | Argentina v France Full Penalty Shoot-out.mp4 | parallel

- Video path: `Videos/Argentina v France Full Penalty Shoot-out.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `241.338` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.548 |
| save_clips | - |
| sample_frames | 6.789 |
| caption_frames | 137.554 |
| sample_fps | 60.641 |
| detect_object_yolo | 31.402 |
| audio_scan | 25.538 |
| asr_timings | 32.443 |
| ast_timings | 15.720 |
| describe_scenes | 65.616 |
| summarize_scenes | 13.217 |
| synthesize_synopsis | 11.118 |
| make_embedding | 4.088 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 144.350 |
| branch_yolo_total | 92.050 |
| branch_audio_total | 57.992 |

## --- Batched BLIP Processing (batches = [1,4,8])---

## 2026-04-06 14:31:01 UTC | Argentina v France Full Penalty Shoot-out.mp4 | parallel

- Video path: `Videos/Argentina v France Full Penalty Shoot-out.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `277.033` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.559 |
| save_clips | - |
| sample_frames | 6.770 |
| caption_frames | 134.673 |
| sample_fps | 60.753 |
| detect_object_yolo | 30.881 |
| audio_scan | 25.785 |
| asr_timings | 32.610 |
| ast_timings | 15.323 |
| describe_scenes | 84.730 |
| summarize_scenes | 30.992 |
| synthesize_synopsis | 12.798 |
| make_embedding | 4.107 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 141.449 |
| branch_yolo_total | 91.641 |
| branch_audio_total | 58.404 |

## 2026-04-06 14:36:51 UTC | Argentina v France Full Penalty Shoot-out.mp4 | parallel

- Video path: `Videos/Argentina v France Full Penalty Shoot-out.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `206.972` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.505 |
| save_clips | - |
| sample_frames | 6.712 |
| caption_frames | 59.180 |
| sample_fps | 68.653 |
| detect_object_yolo | 28.135 |
| audio_scan | 29.175 |
| asr_timings | 33.440 |
| ast_timings | 15.447 |
| describe_scenes | 78.893 |
| summarize_scenes | 14.636 |
| synthesize_synopsis | 9.454 |
| make_embedding | 4.294 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 65.897 |
| branch_yolo_total | 96.794 |
| branch_audio_total | 62.625 |

## 2026-04-06 14:43:02 UTC | Argentina v France Full Penalty Shoot-out.mp4 | parallel

- Video path: `Videos/Argentina v France Full Penalty Shoot-out.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `222.339` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.531 |
| save_clips | - |
| sample_frames | 6.713 |
| caption_frames | 59.361 |
| sample_fps | 68.714 |
| detect_object_yolo | 27.977 |
| audio_scan | 29.119 |
| asr_timings | 34.239 |
| ast_timings | 15.598 |
| describe_scenes | 79.686 |
| summarize_scenes | 26.917 |
| synthesize_synopsis | 12.064 |
| make_embedding | 4.047 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 66.080 |
| branch_yolo_total | 96.699 |
| branch_audio_total | 63.368 |

## --- Implementing a Less Seek-heavy YOLO Frame Sampling Strategy (Sequential) ---

## 2026-04-06 15:39:04 UTC | Argentina v France Full Penalty Shoot-out.mp4 | parallel

- Video path: `Videos/Argentina v France Full Penalty Shoot-out.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `238.962` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.563 |
| save_clips | - |
| sample_frames | 6.729 |
| caption_frames | 92.637 |
| sample_fps | 7.548 |
| detect_object_yolo | 33.163 |
| audio_scan | 43.992 |
| asr_timings | 32.548 |
| ast_timings | 15.235 |
| describe_scenes | 95.045 |
| summarize_scenes | 21.786 |
| synthesize_synopsis | 15.618 |
| make_embedding | 4.177 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 99.372 |
| branch_yolo_total | 40.717 |
| branch_audio_total | 76.549 |

## 2026-04-06 16:16:21 UTC | Argentina v France Full Penalty Shoot-out.mp4 | parallel

- Video path: `Videos/Argentina v France Full Penalty Shoot-out.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `219.780` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.531 |
| save_clips | - |
| sample_frames | 6.785 |
| caption_frames | 92.384 |
| sample_fps | 7.614 |
| detect_object_yolo | 32.955 |
| audio_scan | 44.868 |
| asr_timings | 31.863 |
| ast_timings | 15.319 |
| describe_scenes | 86.275 |
| summarize_scenes | 15.266 |
| synthesize_synopsis | 11.968 |
| make_embedding | 4.158 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 99.176 |
| branch_yolo_total | 40.784 |
| branch_audio_total | 76.740 |

## --- After tuning scene description workers & description + LLM cooldown

### workers=8, cooldown=5

## 2026-04-06 16:52:28 UTC | Argentina v France Full Penalty Shoot-out.mp4 | parallel

- Video path: `Videos/Argentina v France Full Penalty Shoot-out.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `312.253` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.510 |
| save_clips | - |
| sample_frames | 6.813 |
| caption_frames | 93.052 |
| sample_fps | 7.606 |
| detect_object_yolo | 33.302 |
| audio_scan | 44.779 |
| asr_timings | 32.629 |
| ast_timings | 14.999 |
| describe_scenes | 176.435 |
| summarize_scenes | 15.714 |
| synthesize_synopsis | 13.126 |
| make_embedding | 4.187 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 99.872 |
| branch_yolo_total | 40.915 |
| branch_audio_total | 77.418 |


### workers=8, cooldown=1

## 2026-04-06 16:59:03 UTC | Argentina v France Full Penalty Shoot-out.mp4 | parallel

- Video path: `Videos/Argentina v France Full Penalty Shoot-out.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `230.812` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.520 |
| save_clips | - |
| sample_frames | 6.759 |
| caption_frames | 92.679 |
| sample_fps | 7.544 |
| detect_object_yolo | 33.058 |
| audio_scan | 44.530 |
| asr_timings | 31.771 |
| ast_timings | 14.978 |
| describe_scenes | 99.506 |
| summarize_scenes | 15.065 |
| synthesize_synopsis | 9.679 |
| make_embedding | 4.197 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 99.446 |
| branch_yolo_total | 40.608 |
| branch_audio_total | 76.310 |


### workers=10, cooldown=1

## 2026-04-06 17:05:20 UTC | Argentina v France Full Penalty Shoot-out.mp4 | parallel

- Video path: `Videos/Argentina v France Full Penalty Shoot-out.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `219.944` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.491 |
| save_clips | - |
| sample_frames | 6.736 |
| caption_frames | 93.034 |
| sample_fps | 7.648 |
| detect_object_yolo | 33.980 |
| audio_scan | 44.300 |
| asr_timings | 31.834 |
| ast_timings | 15.153 |
| describe_scenes | 81.495 |
| summarize_scenes | 19.445 |
| synthesize_synopsis | 12.225 |
| make_embedding | 4.103 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 99.776 |
| branch_yolo_total | 41.634 |
| branch_audio_total | 76.144 |


### --- One more batch size 4 run just to test again

## 2026-04-06 19:09:50 UTC | Argentina v France Full Penalty Shoot-out.mp4 | parallel

- Video path: `Videos/Argentina v France Full Penalty Shoot-out.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `235.286` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.523 |
| save_clips | - |
| sample_frames | 6.748 |
| caption_frames | 93.565 |
| sample_fps | 7.649 |
| detect_object_yolo | 33.242 |
| audio_scan | 43.521 |
| asr_timings | 32.005 |
| ast_timings | 14.888 |
| describe_scenes | 93.626 |
| summarize_scenes | 22.756 |
| synthesize_synopsis | 11.500 |
| make_embedding | 4.157 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 100.320 |
| branch_yolo_total | 40.897 |
| branch_audio_total | 75.536 |
