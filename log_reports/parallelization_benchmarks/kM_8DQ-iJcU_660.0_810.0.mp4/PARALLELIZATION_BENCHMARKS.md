# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 13:50:52 UTC | kM_8DQ-iJcU_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 151.401 | 0.669 | 32.989 | 34.329 | 15.046 | 26.064 | 3.040 |

## 2026-06-26 13:50:52 UTC | kM_8DQ-iJcU_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/kM_8DQ-iJcU_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `151.401` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.669 |
| save_clips | - |
| sample_frames | 0.812 |
| caption_frames | 32.171 |
| sample_fps | 1.991 |
| detect_object_yolo | 7.513 |
| audio_scan | 3.881 |
| asr_timings | 0.000 |
| ast_timings | 24.419 |
| describe_scenes | 34.329 |
| summarize_scenes | 15.046 |
| synthesize_synopsis | 26.064 |
| make_embedding | 3.040 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.989 |
| branch_yolo_total | 9.511 |
| branch_audio_total | 28.309 |
