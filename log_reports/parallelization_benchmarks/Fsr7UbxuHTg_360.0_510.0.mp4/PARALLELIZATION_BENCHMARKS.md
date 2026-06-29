# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 00:43:30 UTC | Fsr7UbxuHTg_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 136.453 | 0.849 | 50.162 | 12.178 | 9.374 | 8.844 | 3.544 |

## 2026-06-25 00:43:30 UTC | Fsr7UbxuHTg_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Fsr7UbxuHTg_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `136.453` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.849 |
| save_clips | - |
| sample_frames | 1.184 |
| caption_frames | 37.485 |
| sample_fps | 2.303 |
| detect_object_yolo | 9.103 |
| audio_scan | 10.757 |
| asr_timings | 9.911 |
| ast_timings | 29.486 |
| describe_scenes | 12.178 |
| summarize_scenes | 9.374 |
| synthesize_synopsis | 8.844 |
| make_embedding | 3.544 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.674 |
| branch_yolo_total | 11.412 |
| branch_audio_total | 50.162 |
