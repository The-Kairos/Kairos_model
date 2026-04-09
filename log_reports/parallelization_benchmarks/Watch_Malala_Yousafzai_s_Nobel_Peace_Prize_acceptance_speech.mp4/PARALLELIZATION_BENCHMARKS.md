# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-04-08 03:21:00 UTC | Watch_Malala_Yousafzai_s_Nobel_Peace_Prize_acceptance_speech.mp4 | semi_parallel | gemini | gemini-embedding-001 | 163.602 | 2.685 | 108.056 | 24.987 | 9.553 | 11.911 | 1.518 |
| 2026-04-08 03:38:57 UTC | Watch_Malala_Yousafzai_s_Nobel_Peace_Prize_acceptance_speech.mp4 | semi_parallel | gemini | gemini-embedding-001 | 153.462 | 2.649 | 103.711 | 25.983 | 7.869 | 6.799 | 1.488 |
| 2026-04-08 09:39:44 UTC | Watch_Malala_Yousafzai_s_Nobel_Peace_Prize_acceptance_speech.mp4 | parallel | gemini | gemini-embedding-001 | 106.766 | 2.715 | 43.093 | 30.935 | 12.437 | 11.762 | 1.396 |

## 2026-04-08 03:21:00 UTC | Watch_Malala_Yousafzai_s_Nobel_Peace_Prize_acceptance_speech.mp4 | semi_parallel

- Video path: `/var/tmp/kairos/jobs/434e2761-2a61-4f85-9ad7-1030601e63b1/Watch_Malala_Yousafzai_s_Nobel_Peace_Prize_acceptance_speech.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `163.602` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.685 |
| save_clips | - |
| sample_frames | 1.898 |
| caption_frames | 17.802 |
| sample_fps | 9.627 |
| detect_object_yolo | 12.072 |
| audio_scan | 20.110 |
| asr_timings | 12.977 |
| ast_timings | 33.553 |
| describe_scenes | 24.987 |
| summarize_scenes | 9.553 |
| synthesize_synopsis | 11.911 |
| make_embedding | 1.518 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 19.705 |
| branch_yolo_total | 21.704 |
| branch_audio_total | 66.647 |

## 2026-04-08 03:38:57 UTC | Watch_Malala_Yousafzai_s_Nobel_Peace_Prize_acceptance_speech.mp4 | semi_parallel

- Video path: `/var/tmp/kairos/jobs/bbe42da8-2162-409e-901d-bd60cd61df1e/Watch_Malala_Yousafzai_s_Nobel_Peace_Prize_acceptance_speech.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `153.462` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.649 |
| save_clips | - |
| sample_frames | 1.824 |
| caption_frames | 17.064 |
| sample_fps | 9.521 |
| detect_object_yolo | 11.798 |
| audio_scan | 20.096 |
| asr_timings | 11.733 |
| ast_timings | 31.659 |
| describe_scenes | 25.983 |
| summarize_scenes | 7.869 |
| synthesize_synopsis | 6.799 |
| make_embedding | 1.488 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 18.892 |
| branch_yolo_total | 21.324 |
| branch_audio_total | 63.495 |

## 2026-04-08 09:39:44 UTC | Watch_Malala_Yousafzai_s_Nobel_Peace_Prize_acceptance_speech.mp4 | parallel

- Video path: `/var/tmp/kairos/jobs/66f111a8-0f43-45a1-900f-b5ee6edd6e58/Watch_Malala_Yousafzai_s_Nobel_Peace_Prize_acceptance_speech.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `106.766` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.715 |
| save_clips | - |
| sample_frames | 2.326 |
| caption_frames | 26.482 |
| sample_fps | 13.921 |
| detect_object_yolo | 15.397 |
| audio_scan | 30.890 |
| asr_timings | 12.193 |
| ast_timings | 12.192 |
| describe_scenes | 30.935 |
| summarize_scenes | 12.437 |
| synthesize_synopsis | 11.762 |
| make_embedding | 1.396 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 28.815 |
| branch_yolo_total | 29.326 |
| branch_audio_total | 43.093 |
