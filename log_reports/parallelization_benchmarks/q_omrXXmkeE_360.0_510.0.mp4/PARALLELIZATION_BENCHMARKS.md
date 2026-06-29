# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 08:51:22 UTC | q_omrXXmkeE_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 161.581 | 0.791 | 60.005 | 12.907 | 14.595 | 9.526 | 3.868 |

## 2026-06-28 08:51:22 UTC | q_omrXXmkeE_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/q_omrXXmkeE_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `161.581` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.791 |
| save_clips | - |
| sample_frames | 1.557 |
| caption_frames | 44.591 |
| sample_fps | 2.525 |
| detect_object_yolo | 9.765 |
| audio_scan | 12.905 |
| asr_timings | 13.735 |
| ast_timings | 33.357 |
| describe_scenes | 12.907 |
| summarize_scenes | 14.595 |
| synthesize_synopsis | 9.526 |
| make_embedding | 3.868 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.154 |
| branch_yolo_total | 12.295 |
| branch_audio_total | 60.005 |
