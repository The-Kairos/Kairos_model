# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 05:05:37 UTC | J7N2j6leva4_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 99.992 | 0.669 | 38.023 | 7.182 | 6.973 | 15.746 | 2.050 |

## 2026-06-25 05:05:37 UTC | J7N2j6leva4_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/J7N2j6leva4_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `99.992` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.669 |
| save_clips | - |
| sample_frames | 0.465 |
| caption_frames | 18.761 |
| sample_fps | 1.799 |
| detect_object_yolo | 6.905 |
| audio_scan | 12.708 |
| asr_timings | 12.580 |
| ast_timings | 12.726 |
| describe_scenes | 7.182 |
| summarize_scenes | 6.973 |
| synthesize_synopsis | 15.746 |
| make_embedding | 2.050 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 19.232 |
| branch_yolo_total | 8.710 |
| branch_audio_total | 38.023 |
