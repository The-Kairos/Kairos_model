# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-04-07 21:54:06 UTC | How_to_Make_Pasta_-_Without_a_Machine.mp4 | semi_parallel | gemini | gemini-embedding-001 | 317.652 | 2.930 | 223.856 | 59.085 | 14.992 | 7.998 | 3.632 |
| 2026-04-08 03:07:34 UTC | How_to_Make_Pasta_-_Without_a_Machine.mp4 | semi_parallel | gemini | gemini-embedding-001 | 274.540 | 2.907 | 192.943 | 52.495 | 10.015 | 7.490 | 3.462 |

## 2026-04-07 21:54:06 UTC | How_to_Make_Pasta_-_Without_a_Machine.mp4 | semi_parallel

- Video path: `/var/tmp/kairos/jobs/57da22e4-2ce8-49ad-8d0e-af6c34ca867b/How_to_Make_Pasta_-_Without_a_Machine.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `317.652` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.930 |
| save_clips | - |
| sample_frames | 6.173 |
| caption_frames | 46.446 |
| sample_fps | 11.813 |
| detect_object_yolo | 19.752 |
| audio_scan | 21.549 |
| asr_timings | 17.468 |
| ast_timings | 100.638 |
| describe_scenes | 59.085 |
| summarize_scenes | 14.992 |
| synthesize_synopsis | 7.998 |
| make_embedding | 3.632 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 52.624 |
| branch_yolo_total | 31.570 |
| branch_audio_total | 139.662 |

## 2026-04-08 03:07:34 UTC | How_to_Make_Pasta_-_Without_a_Machine.mp4 | semi_parallel

- Video path: `/var/tmp/kairos/jobs/243ef874-e974-4788-be87-0d7eb8570a7d/How_to_Make_Pasta_-_Without_a_Machine.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `False`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `274.540` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.907 |
| save_clips | - |
| sample_frames | 6.018 |
| caption_frames | 35.052 |
| sample_fps | 11.693 |
| detect_object_yolo | 19.354 |
| audio_scan | 21.030 |
| asr_timings | 16.623 |
| ast_timings | 83.156 |
| describe_scenes | 52.495 |
| summarize_scenes | 10.015 |
| synthesize_synopsis | 7.490 |
| make_embedding | 3.462 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.075 |
| branch_yolo_total | 31.051 |
| branch_audio_total | 120.817 |
