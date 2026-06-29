# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 18:50:15 UTC | APwfWItEVks_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 143.133 | 0.790 | 49.924 | 16.077 | 12.279 | 16.650 | 3.096 |

## 2026-06-24 18:50:15 UTC | APwfWItEVks_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/APwfWItEVks_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `143.133` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.790 |
| save_clips | - |
| sample_frames | 0.737 |
| caption_frames | 31.870 |
| sample_fps | 2.136 |
| detect_object_yolo | 8.171 |
| audio_scan | 16.026 |
| asr_timings | 9.500 |
| ast_timings | 24.389 |
| describe_scenes | 16.077 |
| summarize_scenes | 12.279 |
| synthesize_synopsis | 16.650 |
| make_embedding | 3.096 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.613 |
| branch_yolo_total | 10.312 |
| branch_audio_total | 49.924 |
