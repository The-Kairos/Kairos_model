# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 21:03:45 UTC | CrcrPv8Huvs_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 139.275 | 0.666 | 52.109 | 13.323 | 7.713 | 8.328 | 3.564 |

## 2026-06-24 21:03:45 UTC | CrcrPv8Huvs_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/CrcrPv8Huvs_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `139.275` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.666 |
| save_clips | - |
| sample_frames | 1.023 |
| caption_frames | 40.118 |
| sample_fps | 2.088 |
| detect_object_yolo | 8.953 |
| audio_scan | 13.913 |
| asr_timings | 8.867 |
| ast_timings | 29.320 |
| describe_scenes | 13.323 |
| summarize_scenes | 7.713 |
| synthesize_synopsis | 8.328 |
| make_embedding | 3.564 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.146 |
| branch_yolo_total | 11.047 |
| branch_audio_total | 52.109 |
