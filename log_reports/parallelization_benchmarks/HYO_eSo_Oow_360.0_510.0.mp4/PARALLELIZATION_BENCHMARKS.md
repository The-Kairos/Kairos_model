# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 03:42:21 UTC | HYO_eSo_Oow_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 212.962 | 0.823 | 77.401 | 15.322 | 12.201 | 16.116 | 6.516 |

## 2026-06-25 03:42:21 UTC | HYO_eSo_Oow_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/HYO_eSo_Oow_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `212.962` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.823 |
| save_clips | - |
| sample_frames | 1.733 |
| caption_frames | 65.981 |
| sample_fps | 2.731 |
| detect_object_yolo | 12.722 |
| audio_scan | 11.778 |
| asr_timings | 13.841 |
| ast_timings | 51.773 |
| describe_scenes | 15.322 |
| summarize_scenes | 12.201 |
| synthesize_synopsis | 16.116 |
| make_embedding | 6.516 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 67.720 |
| branch_yolo_total | 15.459 |
| branch_audio_total | 77.401 |
