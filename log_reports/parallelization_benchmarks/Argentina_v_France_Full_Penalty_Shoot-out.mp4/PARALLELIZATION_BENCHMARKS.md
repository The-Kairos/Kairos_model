# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-04-07 21:37:43 UTC | Argentina_v_France_Full_Penalty_Shoot-out.mp4 | semi_parallel | gemini | gemini-embedding-001 | 343.951 | 2.563 | 225.986 | 80.753 | 14.275 | 11.155 | 3.951 |
| 2026-04-08 02:43:14 UTC | Argentina_v_France_Full_Penalty_Shoot-out.mp4 | semi_parallel | gemini | gemini-embedding-001 | 325.569 | 2.607 | 226.544 | 66.905 | 12.389 | 7.695 | 4.176 |
| 2026-04-08 09:07:04 UTC | Argentina_v_France_Full_Penalty_Shoot-out.mp4 | parallel | gemini | gemini-embedding-001 | 232.376 | 2.540 | 89.859 | 92.262 | 23.624 | 15.379 | 3.948 |

## 2026-04-07 21:37:43 UTC | Argentina_v_France_Full_Penalty_Shoot-out.mp4 | semi_parallel

- Video path: `/var/tmp/kairos/jobs/b1e2ec55-27cc-4973-aa73-1d73781646e3/Argentina_v_France_Full_Penalty_Shoot-out.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `343.951` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.563 |
| save_clips | - |
| sample_frames | 5.735 |
| caption_frames | 44.342 |
| sample_fps | 6.114 |
| detect_object_yolo | 26.204 |
| audio_scan | 22.606 |
| asr_timings | 20.646 |
| ast_timings | 100.321 |
| describe_scenes | 80.753 |
| summarize_scenes | 14.275 |
| synthesize_synopsis | 11.155 |
| make_embedding | 3.951 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 50.081 |
| branch_yolo_total | 32.323 |
| branch_audio_total | 143.581 |

## 2026-04-08 02:43:14 UTC | Argentina_v_France_Full_Penalty_Shoot-out.mp4 | semi_parallel

- Video path: `/var/tmp/kairos/jobs/d058f058-2304-4373-b978-b1b7ddeca36d/Argentina_v_France_Full_Penalty_Shoot-out.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `325.569` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.607 |
| save_clips | - |
| sample_frames | 5.794 |
| caption_frames | 44.407 |
| sample_fps | 6.106 |
| detect_object_yolo | 26.759 |
| audio_scan | 22.010 |
| asr_timings | 21.274 |
| ast_timings | 100.175 |
| describe_scenes | 66.905 |
| summarize_scenes | 12.389 |
| synthesize_synopsis | 7.695 |
| make_embedding | 4.176 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 50.206 |
| branch_yolo_total | 32.871 |
| branch_audio_total | 143.467 |

## 2026-04-08 09:07:04 UTC | Argentina_v_France_Full_Penalty_Shoot-out.mp4 | parallel

- Video path: `/var/tmp/kairos/jobs/7601f3ad-7c21-4a6d-ae15-efa4a1df7496/Argentina_v_France_Full_Penalty_Shoot-out.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `232.376` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.540 |
| save_clips | - |
| sample_frames | 6.692 |
| caption_frames | 83.159 |
| sample_fps | 7.586 |
| detect_object_yolo | 34.261 |
| audio_scan | 44.054 |
| asr_timings | 23.842 |
| ast_timings | 34.319 |
| describe_scenes | 92.262 |
| summarize_scenes | 23.624 |
| synthesize_synopsis | 15.379 |
| make_embedding | 3.948 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 89.859 |
| branch_yolo_total | 41.855 |
| branch_audio_total | 78.382 |
