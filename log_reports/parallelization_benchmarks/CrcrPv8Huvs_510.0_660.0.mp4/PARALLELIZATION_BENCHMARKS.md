# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 21:01:25 UTC | CrcrPv8Huvs_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 125.182 | 0.710 | 43.505 | 9.740 | 18.872 | 12.845 | 2.629 |

## 2026-06-24 21:01:25 UTC | CrcrPv8Huvs_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/CrcrPv8Huvs_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `125.182` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.710 |
| save_clips | - |
| sample_frames | 0.963 |
| caption_frames | 25.131 |
| sample_fps | 2.036 |
| detect_object_yolo | 7.368 |
| audio_scan | 13.921 |
| asr_timings | 11.238 |
| ast_timings | 18.337 |
| describe_scenes | 9.740 |
| summarize_scenes | 18.872 |
| synthesize_synopsis | 12.845 |
| make_embedding | 2.629 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 26.101 |
| branch_yolo_total | 9.409 |
| branch_audio_total | 43.505 |
