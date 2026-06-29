# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 05:13:45 UTC | yX8oV3E5VSU_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 208.946 | 0.668 | 94.905 | 11.641 | 14.942 | 7.434 | 5.412 |

## 2026-06-27 05:13:45 UTC | yX8oV3E5VSU_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/yX8oV3E5VSU_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `208.946` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.668 |
| save_clips | - |
| sample_frames | 1.820 |
| caption_frames | 57.251 |
| sample_fps | 2.555 |
| detect_object_yolo | 10.902 |
| audio_scan | 13.083 |
| asr_timings | 37.701 |
| ast_timings | 44.112 |
| describe_scenes | 11.641 |
| summarize_scenes | 14.942 |
| synthesize_synopsis | 7.434 |
| make_embedding | 5.412 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 59.076 |
| branch_yolo_total | 13.463 |
| branch_audio_total | 94.905 |
