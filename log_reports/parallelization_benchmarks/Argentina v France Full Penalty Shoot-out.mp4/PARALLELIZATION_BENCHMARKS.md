# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-04-06 07:13:01 UTC | Argentina v France Full Penalty Shoot-out.mp4 | semi_parallel | gemini | gemini-embedding-001 | 317.074 | 2.497 | 84.340 | 79.458 | 13.840 | 8.087 | 3.853 |
| 2026-04-06 07:17:21 UTC | Argentina v France Full Penalty Shoot-out.mp4 | parallel | gemini | gemini-embedding-001 | 246.552 | 2.476 | 140.177 | 74.794 | 12.942 | 12.055 | 3.940 |
| 2026-04-06 11:20:23 UTC | Argentina v France Full Penalty Shoot-out.mp4 | parallel | gemini | gemini-embedding-001 | 241.338 | 2.548 | 144.350 | 65.616 | 13.217 | 11.118 | 4.088 |
| 2026-04-06 14:31:01 UTC | Argentina v France Full Penalty Shoot-out.mp4 | parallel | gemini | gemini-embedding-001 | 277.033 | 2.559 | 141.449 | 84.730 | 30.992 | 12.798 | 4.107 |
| 2026-04-06 14:36:51 UTC | Argentina v France Full Penalty Shoot-out.mp4 | parallel | gemini | gemini-embedding-001 | 206.972 | 2.505 | 96.794 | 78.893 | 14.636 | 9.454 | 4.294 |
| 2026-04-06 14:43:02 UTC | Argentina v France Full Penalty Shoot-out.mp4 | parallel | gemini | gemini-embedding-001 | 222.339 | 2.531 | 96.699 | 79.686 | 26.917 | 12.064 | 4.047 |


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
