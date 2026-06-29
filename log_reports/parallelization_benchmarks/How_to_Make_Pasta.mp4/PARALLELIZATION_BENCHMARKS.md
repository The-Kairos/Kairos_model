# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-04-11 17:01:22 UTC | How_to_Make_Pasta.mp4 | parallel | gemini | gemini-embedding-001 | 202.578 | 2.904 | 137.661 | 37.288 | 8.009 | 8.190 | 3.461 |

## 2026-04-11 17:01:22 UTC | How_to_Make_Pasta.mp4 | parallel

- Video path: `/home/Kairos_model/.tmp/kairos/jobs/4eac8641-4cec-41de-8b32-52c3640fc166/How_to_Make_Pasta.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `202.578` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 2.904 |
| save_clips | - |
| sample_frames | 7.299 |
| caption_frames | 66.372 |
| sample_fps | 15.496 |
| detect_object_yolo | 25.746 |
| audio_scan | 33.044 |
| asr_timings | 23.599 |
| ast_timings | 104.608 |
| describe_scenes | 37.288 |
| summarize_scenes | 8.009 |
| synthesize_synopsis | 8.190 |
| make_embedding | 3.461 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 73.678 |
| branch_yolo_total | 41.250 |
| branch_audio_total | 137.661 |
