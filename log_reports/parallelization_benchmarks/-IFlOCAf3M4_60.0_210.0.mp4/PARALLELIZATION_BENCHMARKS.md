# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 07:59:29 UTC | -IFlOCAf3M4_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 232.029 | 0.844 | 116.173 | 14.655 | 14.518 | 19.220 | 4.297 |

## 2026-06-24 07:59:29 UTC | -IFlOCAf3M4_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-IFlOCAf3M4_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `232.029` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.844 |
| save_clips | - |
| sample_frames | 1.446 |
| caption_frames | 47.420 |
| sample_fps | 2.398 |
| detect_object_yolo | 9.625 |
| audio_scan | 14.844 |
| asr_timings | 65.899 |
| ast_timings | 35.421 |
| describe_scenes | 14.655 |
| summarize_scenes | 14.518 |
| synthesize_synopsis | 19.220 |
| make_embedding | 4.297 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 48.871 |
| branch_yolo_total | 12.029 |
| branch_audio_total | 116.173 |
