# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 03:14:50 UTC | HAttn-5lM3Y_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 164.807 | 0.675 | 54.729 | 16.247 | 12.220 | 16.203 | 3.831 |

## 2026-06-25 03:14:50 UTC | HAttn-5lM3Y_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/HAttn-5lM3Y_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `164.807` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.675 |
| save_clips | - |
| sample_frames | 1.174 |
| caption_frames | 46.561 |
| sample_fps | 2.213 |
| detect_object_yolo | 9.532 |
| audio_scan | 13.776 |
| asr_timings | 9.434 |
| ast_timings | 31.510 |
| describe_scenes | 16.247 |
| summarize_scenes | 12.220 |
| synthesize_synopsis | 16.203 |
| make_embedding | 3.831 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.741 |
| branch_yolo_total | 11.751 |
| branch_audio_total | 54.729 |
