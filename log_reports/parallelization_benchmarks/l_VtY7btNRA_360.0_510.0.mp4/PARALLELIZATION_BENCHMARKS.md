# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 15:39:19 UTC | l_VtY7btNRA_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 177.163 | 0.789 | 46.410 | 11.477 | 49.473 | 24.304 | 2.778 |

## 2026-06-26 15:39:19 UTC | l_VtY7btNRA_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/l_VtY7btNRA_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `177.163` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.789 |
| save_clips | - |
| sample_frames | 0.660 |
| caption_frames | 29.913 |
| sample_fps | 2.116 |
| detect_object_yolo | 7.820 |
| audio_scan | 12.969 |
| asr_timings | 11.711 |
| ast_timings | 21.721 |
| describe_scenes | 11.477 |
| summarize_scenes | 49.473 |
| synthesize_synopsis | 24.304 |
| make_embedding | 2.778 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.578 |
| branch_yolo_total | 9.942 |
| branch_audio_total | 46.410 |
