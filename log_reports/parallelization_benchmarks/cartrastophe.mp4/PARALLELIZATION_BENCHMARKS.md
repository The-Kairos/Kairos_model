# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-04-08 07:28:05 UTC | cartrastophe.mp4 | semi_parallel | gemini | gemini-embedding-001 | 209.028 | 2.591 | 133.180 | 46.701 | 11.788 | 7.566 | 2.387 |
| 2026-04-08 07:43:22 UTC | cartrastophe.mp4 | parallel | gemini | gemini-embedding-001 | 309.462 | 2.740 | 172.778 | 105.780 | 11.315 | 9.385 | 2.477 |
| 2026-04-08 07:49:43 UTC | cartrastophe.mp4 | parallel | gemini | gemini-embedding-001 | 158.408 | 2.619 | 75.374 | 46.531 | 16.006 | 10.998 | 2.405 |
| 2026-04-08 07:53:51 UTC | cartrastophe.mp4 | parallel | gemini | gemini-embedding-001 | 155.996 | 2.543 | 74.355 | 43.596 | 16.273 | 12.449 | 2.411 |
| 2026-04-08 07:59:37 UTC | cartrastophe.mp4 | parallel | gemini | gemini-embedding-001 | 158.928 | 2.619 | 75.095 | 45.687 | 16.598 | 12.021 | 2.468 |
| 2026-04-08 08:04:21 UTC | cartrastophe.mp4 | parallel | gemini | gemini-embedding-001 | 169.555 | 2.636 | 74.394 | 52.641 | 18.491 | 14.400 | 2.525 |
| 2026-04-08 08:14:55 UTC | cartrastophe.mp4 | parallel | gemini | gemini-embedding-001 | 172.513 | 2.662 | 74.424 | 55.181 | 14.208 | 19.050 | 2.461 |
| 2026-04-08 08:24:12 UTC | cartrastophe.mp4 | parallel | gemini | gemini-embedding-001 | 145.957 | 2.903 | 50.036 | 50.073 | 21.299 | 14.258 | 2.529 |
| 2026-04-08 08:35:47 UTC | cartrastophe.mp4 | parallel | gemini | gemini-embedding-001 | 135.017 | 2.643 | 45.677 | 49.089 | 18.888 | 11.498 | 2.561 |
| 2026-04-08 08:41:03 UTC | cartrastophe.mp4 | parallel | gemini | gemini-embedding-001 | 128.727 | 2.779 | 45.564 | 46.955 | 16.708 | 9.900 | 2.391 |
| 2026-04-08 08:45:06 UTC | cartrastophe.mp4 | parallel | gemini | gemini-embedding-001 | 124.844 | 2.642 | 45.177 | 45.663 | 13.444 | 10.930 | 2.535 |
| 2026-04-08 08:55:46 UTC | cartrastophe.mp4 | parallel | gemini | gemini-embedding-001 | 136.015 | 2.836 | 51.662 | 47.293 | 17.413 | 9.755 | 2.457 |
| 2026-04-08 08:58:48 UTC | cartrastophe.mp4 | parallel | gemini | gemini-embedding-001 | 123.317 | 2.690 | 46.105 | 43.785 | 14.705 | 9.180 | 2.402 |
| 2026-04-08 10:51:03 UTC | cartrastophe.mp4 | parallel | gemini | gemini-embedding-001 | 136.337 | 2.819 | 50.317 | 46.263 | 17.688 | 11.858 | 2.599 |

## 2026-04-08 07:28:05 UTC | cartrastophe.mp4 | semi_parallel

- Video path: `/var/tmp/kairos/jobs/12da969f-6de9-4f3e-bccf-de3a760d1e3e/cartrastophe.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `209.028` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.591 |
| save_clips | - |
| sample_frames | 4.233 |
| caption_frames | 25.835 |
| sample_fps | 9.852 |
| detect_object_yolo | 13.990 |
| audio_scan | 17.105 |
| asr_timings | 14.346 |
| ast_timings | 47.802 |
| describe_scenes | 46.701 |
| summarize_scenes | 11.788 |
| synthesize_synopsis | 7.566 |
| make_embedding | 2.387 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.073 |
| branch_yolo_total | 23.847 |
| branch_audio_total | 79.260 |

## 2026-04-08 07:43:22 UTC | cartrastophe.mp4 | parallel

- Video path: `/var/tmp/kairos/jobs/0d31f5a0-feac-4f20-92bb-7300bb0fed55/cartrastophe.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `309.462` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.740 |
| save_clips | - |
| sample_frames | 5.818 |
| caption_frames | 166.953 |
| sample_fps | 13.638 |
| detect_object_yolo | 14.439 |
| audio_scan | 29.751 |
| asr_timings | 15.024 |
| ast_timings | 79.990 |
| describe_scenes | 105.780 |
| summarize_scenes | 11.315 |
| synthesize_synopsis | 9.385 |
| make_embedding | 2.477 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 172.778 |
| branch_yolo_total | 28.086 |
| branch_audio_total | 109.750 |

## 2026-04-08 07:49:43 UTC | cartrastophe.mp4 | parallel

- Video path: `/var/tmp/kairos/jobs/08addf5e-2338-4e69-b59f-842718e72b32/cartrastophe.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `158.408` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.619 |
| save_clips | - |
| sample_frames | 5.242 |
| caption_frames | 37.834 |
| sample_fps | 13.406 |
| detect_object_yolo | 17.043 |
| audio_scan | 28.273 |
| asr_timings | 16.960 |
| ast_timings | 47.089 |
| describe_scenes | 46.531 |
| summarize_scenes | 16.006 |
| synthesize_synopsis | 10.998 |
| make_embedding | 2.405 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.081 |
| branch_yolo_total | 30.457 |
| branch_audio_total | 75.374 |

## 2026-04-08 07:53:51 UTC | cartrastophe.mp4 | parallel

- Video path: `/var/tmp/kairos/jobs/bee4f589-ebd2-4532-828b-13bc63ed767b/cartrastophe.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `155.996` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.543 |
| save_clips | - |
| sample_frames | 5.246 |
| caption_frames | 37.345 |
| sample_fps | 13.333 |
| detect_object_yolo | 17.986 |
| audio_scan | 28.039 |
| asr_timings | 18.706 |
| ast_timings | 46.303 |
| describe_scenes | 43.596 |
| summarize_scenes | 16.273 |
| synthesize_synopsis | 12.449 |
| make_embedding | 2.411 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 42.598 |
| branch_yolo_total | 31.327 |
| branch_audio_total | 74.355 |

## 2026-04-08 07:59:37 UTC | cartrastophe.mp4 | parallel

- Video path: `/var/tmp/kairos/jobs/2c9ddb85-31a2-4c64-a092-96d52ef0d020/cartrastophe.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `158.928` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.619 |
| save_clips | - |
| sample_frames | 5.332 |
| caption_frames | 38.291 |
| sample_fps | 13.048 |
| detect_object_yolo | 17.265 |
| audio_scan | 28.492 |
| asr_timings | 16.658 |
| ast_timings | 46.593 |
| describe_scenes | 45.687 |
| summarize_scenes | 16.598 |
| synthesize_synopsis | 12.021 |
| make_embedding | 2.468 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.630 |
| branch_yolo_total | 30.321 |
| branch_audio_total | 75.095 |

## 2026-04-08 08:04:21 UTC | cartrastophe.mp4 | parallel

- Video path: `/var/tmp/kairos/jobs/6c616d88-4b92-4c2e-8f5a-1c524fd44277/cartrastophe.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `169.555` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.636 |
| save_clips | - |
| sample_frames | 5.283 |
| caption_frames | 38.412 |
| sample_fps | 12.849 |
| detect_object_yolo | 17.436 |
| audio_scan | 27.997 |
| asr_timings | 16.967 |
| ast_timings | 46.388 |
| describe_scenes | 52.641 |
| summarize_scenes | 18.491 |
| synthesize_synopsis | 14.400 |
| make_embedding | 2.525 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.703 |
| branch_yolo_total | 30.292 |
| branch_audio_total | 74.394 |

## 2026-04-08 08:14:55 UTC | cartrastophe.mp4 | parallel

- Video path: `/var/tmp/kairos/jobs/024a7bc2-6365-43e8-8ed1-7dc2937738b1/cartrastophe.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `172.513` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.662 |
| save_clips | - |
| sample_frames | 5.272 |
| caption_frames | 40.847 |
| sample_fps | 14.896 |
| detect_object_yolo | 17.917 |
| audio_scan | 28.258 |
| asr_timings | 18.937 |
| ast_timings | 46.155 |
| describe_scenes | 55.181 |
| summarize_scenes | 14.208 |
| synthesize_synopsis | 19.050 |
| make_embedding | 2.461 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.127 |
| branch_yolo_total | 32.820 |
| branch_audio_total | 74.424 |

## 2026-04-08 08:24:12 UTC | cartrastophe.mp4 | parallel

- Video path: `/var/tmp/kairos/jobs/786fe4cc-9506-4501-947e-208ee9f5f38c/cartrastophe.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `145.957` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.903 |
| save_clips | - |
| sample_frames | 5.658 |
| caption_frames | 42.196 |
| sample_fps | 15.649 |
| detect_object_yolo | 16.066 |
| audio_scan | 33.041 |
| asr_timings | 16.494 |
| ast_timings | 16.987 |
| describe_scenes | 50.073 |
| summarize_scenes | 21.299 |
| synthesize_synopsis | 14.258 |
| make_embedding | 2.529 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.860 |
| branch_yolo_total | 31.726 |
| branch_audio_total | 50.036 |

## 2026-04-08 08:35:47 UTC | cartrastophe.mp4 | parallel

- Video path: `/var/tmp/kairos/jobs/01898079-2aee-4e03-a112-ed6e4cead868/cartrastophe.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `135.017` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.643 |
| save_clips | - |
| sample_frames | 5.298 |
| caption_frames | 39.908 |
| sample_fps | 13.277 |
| detect_object_yolo | 18.008 |
| audio_scan | 27.456 |
| asr_timings | 18.210 |
| ast_timings | 17.166 |
| describe_scenes | 49.089 |
| summarize_scenes | 18.888 |
| synthesize_synopsis | 11.498 |
| make_embedding | 2.561 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.212 |
| branch_yolo_total | 31.293 |
| branch_audio_total | 45.677 |

## 2026-04-08 08:41:03 UTC | cartrastophe.mp4 | parallel

- Video path: `/var/tmp/kairos/jobs/ca389def-b461-49a5-b9de-e35b4e932623/cartrastophe.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `128.727` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.779 |
| save_clips | - |
| sample_frames | 5.282 |
| caption_frames | 40.276 |
| sample_fps | 12.912 |
| detect_object_yolo | 17.959 |
| audio_scan | 28.308 |
| asr_timings | 16.888 |
| ast_timings | 16.961 |
| describe_scenes | 46.955 |
| summarize_scenes | 16.708 |
| synthesize_synopsis | 9.900 |
| make_embedding | 2.391 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.564 |
| branch_yolo_total | 30.881 |
| branch_audio_total | 45.280 |

## 2026-04-08 08:45:06 UTC | cartrastophe.mp4 | parallel

- Video path: `/var/tmp/kairos/jobs/59bdc5c8-8efd-488c-8815-a94101e9f83d/cartrastophe.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `124.844` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.642 |
| save_clips | - |
| sample_frames | 5.253 |
| caption_frames | 39.296 |
| sample_fps | 13.296 |
| detect_object_yolo | 17.841 |
| audio_scan | 27.364 |
| asr_timings | 17.804 |
| ast_timings | 17.258 |
| describe_scenes | 45.663 |
| summarize_scenes | 13.444 |
| synthesize_synopsis | 10.930 |
| make_embedding | 2.535 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.556 |
| branch_yolo_total | 31.147 |
| branch_audio_total | 45.177 |

## 2026-04-08 08:55:46 UTC | cartrastophe.mp4 | parallel

- Video path: `/var/tmp/kairos/jobs/32f2bd0d-a983-40f2-8fce-a5278d72545a/cartrastophe.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `136.015` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.836 |
| save_clips | - |
| sample_frames | 5.591 |
| caption_frames | 42.062 |
| sample_fps | 18.105 |
| detect_object_yolo | 17.363 |
| audio_scan | 34.672 |
| asr_timings | 16.817 |
| ast_timings | 16.981 |
| describe_scenes | 47.293 |
| summarize_scenes | 17.413 |
| synthesize_synopsis | 9.755 |
| make_embedding | 2.457 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.660 |
| branch_yolo_total | 35.478 |
| branch_audio_total | 51.662 |

## 2026-04-08 08:58:48 UTC | cartrastophe.mp4 | parallel

- Video path: `/var/tmp/kairos/jobs/c8000e04-13c4-4a7a-8a52-37a680f2b091/cartrastophe.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `123.317` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.690 |
| save_clips | - |
| sample_frames | 5.290 |
| caption_frames | 38.453 |
| sample_fps | 13.450 |
| detect_object_yolo | 17.735 |
| audio_scan | 28.761 |
| asr_timings | 17.331 |
| ast_timings | 17.335 |
| describe_scenes | 43.785 |
| summarize_scenes | 14.705 |
| synthesize_synopsis | 9.180 |
| make_embedding | 2.402 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.749 |
| branch_yolo_total | 31.192 |
| branch_audio_total | 46.105 |

## 2026-04-08 10:51:03 UTC | cartrastophe.mp4 | parallel

- Video path: `/var/tmp/kairos/jobs/1f06dcd8-342e-4398-902a-7cfef80d5e86/cartrastophe.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `136.337` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.819 |
| save_clips | - |
| sample_frames | 5.746 |
| caption_frames | 40.663 |
| sample_fps | 15.975 |
| detect_object_yolo | 16.259 |
| audio_scan | 33.248 |
| asr_timings | 16.483 |
| ast_timings | 17.062 |
| describe_scenes | 46.263 |
| summarize_scenes | 17.688 |
| synthesize_synopsis | 11.858 |
| make_embedding | 2.599 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.418 |
| branch_yolo_total | 32.244 |
| branch_audio_total | 50.317 |
