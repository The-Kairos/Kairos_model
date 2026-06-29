# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 07:43:09 UTC | ooJUCHRH0OM_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 130.897 | 0.799 | 50.702 | 8.207 | 11.561 | 8.061 | 3.224 |

## 2026-06-28 07:43:09 UTC | ooJUCHRH0OM_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ooJUCHRH0OM_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `130.897` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.799 |
| save_clips | - |
| sample_frames | 1.005 |
| caption_frames | 35.418 |
| sample_fps | 2.244 |
| detect_object_yolo | 8.282 |
| audio_scan | 14.803 |
| asr_timings | 9.460 |
| ast_timings | 26.431 |
| describe_scenes | 8.207 |
| summarize_scenes | 11.561 |
| synthesize_synopsis | 8.061 |
| make_embedding | 3.224 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.429 |
| branch_yolo_total | 10.532 |
| branch_audio_total | 50.702 |
