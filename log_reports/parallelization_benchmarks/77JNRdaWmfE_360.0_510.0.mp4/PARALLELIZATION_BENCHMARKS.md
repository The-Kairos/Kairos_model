# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 13:32:48 UTC | 77JNRdaWmfE_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 120.998 | 1.557 | 40.176 | 9.984 | 15.525 | 20.255 | 2.135 |

## 2026-06-24 13:32:48 UTC | 77JNRdaWmfE_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/77JNRdaWmfE_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `120.998` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.557 |
| save_clips | - |
| sample_frames | 0.546 |
| caption_frames | 20.352 |
| sample_fps | 1.963 |
| detect_object_yolo | 7.034 |
| audio_scan | 16.025 |
| asr_timings | 10.210 |
| ast_timings | 13.932 |
| describe_scenes | 9.984 |
| summarize_scenes | 15.525 |
| synthesize_synopsis | 20.255 |
| make_embedding | 2.135 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 20.905 |
| branch_yolo_total | 9.003 |
| branch_audio_total | 40.176 |
