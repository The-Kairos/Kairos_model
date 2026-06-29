# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-04-09 15:41:23 UTC | 5_Mins_Documentary_on_Zebras___Wildlife_Documentary.mp4 | parallel | gemini | gemini-embedding-001 | 133.625 | 3.962 | 55.331 | 40.716 | 16.175 | 11.010 | 1.469 |
| 2026-04-09 16:42:05 UTC | 5_Mins_Documentary_on_Zebras___Wildlife_Documentary.mp4 | parallel | gemini | gemini-embedding-001 | 126.479 | 3.956 | 55.990 | 38.355 | 11.964 | 9.857 | 1.481 |
| 2026-04-10 06:18:54 UTC | 5_Mins_Documentary_on_Zebras___Wildlife_Documentary.mp4 | parallel | gemini | gemini-embedding-001 | 127.636 | 3.912 | 56.371 | 34.256 | 16.538 | 10.204 | 1.439 |
| 2026-04-10 09:12:19 UTC | 5_Mins_Documentary_on_Zebras___Wildlife_Documentary.mp4 | parallel | gemini | gemini-embedding-001 | 138.815 | 4.324 | 62.201 | 33.131 | 16.048 | 16.338 | 1.615 |
| 2026-04-10 09:31:48 UTC | 5_Mins_Documentary_on_Zebras___Wildlife_Documentary.mp4 | parallel | gemini | gemini-embedding-001 | 132.245 | 3.961 | 55.475 | 34.385 | 18.619 | 13.109 | 1.381 |
| 2026-04-10 09:34:12 UTC | 5_Mins_Documentary_on_Zebras___Wildlife_Documentary.mp4 | parallel | gemini | gemini-embedding-001 | 127.140 | 3.954 | 57.584 | 31.474 | 13.317 | 14.518 | 1.436 |

## 2026-04-09 15:41:23 UTC | 5_Mins_Documentary_on_Zebras___Wildlife_Documentary.mp4 | parallel

- Video path: `/home/Kairos_model/.tmp/kairos/jobs/c61e8b3e-d88d-4538-8ac4-d14eebb09e11/5_Mins_Documentary_on_Zebras___Wildlife_Documentary.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `133.625` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 3.962 |
| save_clips | - |
| sample_frames | 4.757 |
| caption_frames | 29.707 |
| sample_fps | 13.543 |
| detect_object_yolo | 17.657 |
| audio_scan | 24.448 |
| asr_timings | 15.730 |
| ast_timings | 30.873 |
| describe_scenes | 40.716 |
| summarize_scenes | 16.175 |
| synthesize_synopsis | 11.010 |
| make_embedding | 1.469 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 34.471 |
| branch_yolo_total | 31.208 |
| branch_audio_total | 55.331 |

## 2026-04-09 16:42:05 UTC | 5_Mins_Documentary_on_Zebras___Wildlife_Documentary.mp4 | parallel

- Video path: `/home/Kairos_model/.tmp/kairos/jobs/d449adf8-d816-4d57-8a2e-0f3fcd40d998/5_Mins_Documentary_on_Zebras___Wildlife_Documentary.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `126.479` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 3.956 |
| save_clips | - |
| sample_frames | 4.786 |
| caption_frames | 28.912 |
| sample_fps | 13.410 |
| detect_object_yolo | 17.031 |
| audio_scan | 25.264 |
| asr_timings | 17.575 |
| ast_timings | 30.718 |
| describe_scenes | 38.355 |
| summarize_scenes | 11.964 |
| synthesize_synopsis | 9.857 |
| make_embedding | 1.481 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 33.707 |
| branch_yolo_total | 30.449 |
| branch_audio_total | 55.990 |

## 2026-04-10 06:18:54 UTC | 5_Mins_Documentary_on_Zebras___Wildlife_Documentary.mp4 | parallel

- Video path: `/home/Kairos_model/.tmp/kairos/jobs/49ac6be0-1049-4009-9939-b51d99b96356/5_Mins_Documentary_on_Zebras___Wildlife_Documentary.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `127.636` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 3.912 |
| save_clips | - |
| sample_frames | 4.647 |
| caption_frames | 28.037 |
| sample_fps | 13.369 |
| detect_object_yolo | 17.747 |
| audio_scan | 25.193 |
| asr_timings | 16.711 |
| ast_timings | 31.169 |
| describe_scenes | 34.256 |
| summarize_scenes | 16.538 |
| synthesize_synopsis | 10.204 |
| make_embedding | 1.439 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.691 |
| branch_yolo_total | 31.125 |
| branch_audio_total | 56.371 |

## 2026-04-10 09:12:19 UTC | 5_Mins_Documentary_on_Zebras___Wildlife_Documentary.mp4 | parallel

- Video path: `/home/Kairos_model/.tmp/kairos/jobs/afb93a6e-65b4-4ccc-8a14-10730e3a48f3/5_Mins_Documentary_on_Zebras___Wildlife_Documentary.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `138.815` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 4.324 |
| save_clips | - |
| sample_frames | 4.993 |
| caption_frames | 31.541 |
| sample_fps | 15.328 |
| detect_object_yolo | 16.612 |
| audio_scan | 31.367 |
| asr_timings | 14.374 |
| ast_timings | 30.822 |
| describe_scenes | 33.131 |
| summarize_scenes | 16.048 |
| synthesize_synopsis | 16.338 |
| make_embedding | 1.615 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.540 |
| branch_yolo_total | 31.948 |
| branch_audio_total | 62.201 |

## 2026-04-10 09:31:48 UTC | 5_Mins_Documentary_on_Zebras___Wildlife_Documentary.mp4 | parallel

- Video path: `/home/Kairos_model/.tmp/kairos/jobs/00744852-3408-4434-a238-e71d4715a63e/5_Mins_Documentary_on_Zebras___Wildlife_Documentary.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `132.245` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 3.961 |
| save_clips | - |
| sample_frames | 4.710 |
| caption_frames | 29.103 |
| sample_fps | 13.446 |
| detect_object_yolo | 17.529 |
| audio_scan | 24.865 |
| asr_timings | 17.775 |
| ast_timings | 30.600 |
| describe_scenes | 34.385 |
| summarize_scenes | 18.619 |
| synthesize_synopsis | 13.109 |
| make_embedding | 1.381 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 33.819 |
| branch_yolo_total | 30.983 |
| branch_audio_total | 55.475 |

## 2026-04-10 09:34:12 UTC | 5_Mins_Documentary_on_Zebras___Wildlife_Documentary.mp4 | parallel

- Video path: `/home/Kairos_model/.tmp/kairos/jobs/52c5e254-59b8-4b9b-aced-2023a5f5910f/5_Mins_Documentary_on_Zebras___Wildlife_Documentary.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `127.140` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 3.954 |
| save_clips | - |
| sample_frames | 4.754 |
| caption_frames | 28.722 |
| sample_fps | 12.905 |
| detect_object_yolo | 17.640 |
| audio_scan | 26.591 |
| asr_timings | 15.462 |
| ast_timings | 30.982 |
| describe_scenes | 31.474 |
| summarize_scenes | 13.317 |
| synthesize_synopsis | 14.518 |
| make_embedding | 1.436 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 33.484 |
| branch_yolo_total | 30.553 |
| branch_audio_total | 57.584 |
