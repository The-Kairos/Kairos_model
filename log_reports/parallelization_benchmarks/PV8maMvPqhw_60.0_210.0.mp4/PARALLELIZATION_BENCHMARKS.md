# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 14:07:34 UTC | PV8maMvPqhw_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 233.374 | 0.638 | 60.363 | 19.935 | 59.337 | 25.722 | 4.264 |

## 2026-06-25 14:07:34 UTC | PV8maMvPqhw_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/PV8maMvPqhw_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `233.374` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.638 |
| save_clips | - |
| sample_frames | 1.598 |
| caption_frames | 47.478 |
| sample_fps | 2.304 |
| detect_object_yolo | 10.247 |
| audio_scan | 15.741 |
| asr_timings | 8.829 |
| ast_timings | 35.784 |
| describe_scenes | 19.935 |
| summarize_scenes | 59.337 |
| synthesize_synopsis | 25.722 |
| make_embedding | 4.264 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.082 |
| branch_yolo_total | 12.557 |
| branch_audio_total | 60.363 |
