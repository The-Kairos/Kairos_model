# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 05:10:15 UTC | yX8oV3E5VSU_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 122.557 | 0.640 | 47.303 | 12.470 | 8.120 | 6.700 | 3.019 |

## 2026-06-27 05:10:15 UTC | yX8oV3E5VSU_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/yX8oV3E5VSU_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `122.557` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.640 |
| save_clips | - |
| sample_frames | 0.884 |
| caption_frames | 32.160 |
| sample_fps | 1.989 |
| detect_object_yolo | 7.814 |
| audio_scan | 12.996 |
| asr_timings | 10.438 |
| ast_timings | 23.860 |
| describe_scenes | 12.470 |
| summarize_scenes | 8.120 |
| synthesize_synopsis | 6.700 |
| make_embedding | 3.019 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 33.050 |
| branch_yolo_total | 9.809 |
| branch_audio_total | 47.303 |
