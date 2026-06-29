# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 08:37:05 UTC | iJRwe2oCX0E_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 128.566 | 0.787 | 39.262 | 16.874 | 15.180 | 20.319 | 2.356 |

## 2026-06-26 08:37:05 UTC | iJRwe2oCX0E_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/iJRwe2oCX0E_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `128.566` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.787 |
| save_clips | - |
| sample_frames | 0.503 |
| caption_frames | 22.588 |
| sample_fps | 1.937 |
| detect_object_yolo | 7.325 |
| audio_scan | 12.937 |
| asr_timings | 10.302 |
| ast_timings | 16.015 |
| describe_scenes | 16.874 |
| summarize_scenes | 15.180 |
| synthesize_synopsis | 20.319 |
| make_embedding | 2.356 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 23.097 |
| branch_yolo_total | 9.268 |
| branch_audio_total | 39.262 |
