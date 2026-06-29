# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 19:23:09 UTC | UqMooNqP7Hs_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 172.756 | 0.804 | 59.203 | 18.686 | 12.347 | 16.740 | 4.080 |

## 2026-06-25 19:23:09 UTC | UqMooNqP7Hs_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/UqMooNqP7Hs_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `172.756` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.804 |
| save_clips | - |
| sample_frames | 1.310 |
| caption_frames | 45.389 |
| sample_fps | 2.392 |
| detect_object_yolo | 10.314 |
| audio_scan | 14.105 |
| asr_timings | 8.934 |
| ast_timings | 36.156 |
| describe_scenes | 18.686 |
| summarize_scenes | 12.347 |
| synthesize_synopsis | 16.740 |
| make_embedding | 4.080 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.705 |
| branch_yolo_total | 12.712 |
| branch_audio_total | 59.203 |
