# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 14:41:20 UTC | Psbtq8LUdqY_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 222.056 | 0.787 | 64.999 | 29.578 | 24.500 | 20.546 | 5.427 |

## 2026-06-25 14:41:20 UTC | Psbtq8LUdqY_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Psbtq8LUdqY_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `222.056` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.787 |
| save_clips | - |
| sample_frames | 1.413 |
| caption_frames | 59.687 |
| sample_fps | 2.539 |
| detect_object_yolo | 11.135 |
| audio_scan | 11.064 |
| asr_timings | 10.649 |
| ast_timings | 43.278 |
| describe_scenes | 29.578 |
| summarize_scenes | 24.500 |
| synthesize_synopsis | 20.546 |
| make_embedding | 5.427 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 61.105 |
| branch_yolo_total | 13.680 |
| branch_audio_total | 64.999 |
