# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 16:38:23 UTC | nUWCO8U02U4_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 81.128 | 0.893 | 24.077 | 8.175 | 5.539 | 6.837 | 2.556 |

## 2026-06-27 16:38:23 UTC | nUWCO8U02U4_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/nUWCO8U02U4_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `81.128` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.893 |
| save_clips | - |
| sample_frames | 0.489 |
| caption_frames | 23.582 |
| sample_fps | 1.994 |
| detect_object_yolo | 7.390 |
| audio_scan | 3.810 |
| asr_timings | 0.000 |
| ast_timings | 18.472 |
| describe_scenes | 8.175 |
| summarize_scenes | 5.539 |
| synthesize_synopsis | 6.837 |
| make_embedding | 2.556 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 24.077 |
| branch_yolo_total | 9.390 |
| branch_audio_total | 22.291 |
