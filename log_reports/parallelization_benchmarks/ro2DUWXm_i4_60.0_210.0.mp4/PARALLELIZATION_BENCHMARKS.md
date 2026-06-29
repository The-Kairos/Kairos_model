# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 19:01:25 UTC | ro2DUWXm_i4_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 142.448 | 0.845 | 49.073 | 13.622 | 14.401 | 15.469 | 3.018 |

## 2026-06-26 19:01:25 UTC | ro2DUWXm_i4_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ro2DUWXm_i4_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `142.448` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.845 |
| save_clips | - |
| sample_frames | 0.917 |
| caption_frames | 33.208 |
| sample_fps | 2.159 |
| detect_object_yolo | 8.291 |
| audio_scan | 14.027 |
| asr_timings | 10.440 |
| ast_timings | 24.599 |
| describe_scenes | 13.622 |
| summarize_scenes | 14.401 |
| synthesize_synopsis | 15.469 |
| make_embedding | 3.018 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 34.131 |
| branch_yolo_total | 10.457 |
| branch_audio_total | 49.073 |
