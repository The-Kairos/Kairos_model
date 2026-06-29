# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 16:42:47 UTC | 8ETILC-7U1w_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 161.801 | 0.667 | 46.041 | 17.457 | 22.969 | 26.473 | 3.046 |

## 2026-06-24 16:42:47 UTC | 8ETILC-7U1w_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/8ETILC-7U1w_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `161.801` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.667 |
| save_clips | - |
| sample_frames | 0.914 |
| caption_frames | 32.719 |
| sample_fps | 2.031 |
| detect_object_yolo | 8.066 |
| audio_scan | 13.809 |
| asr_timings | 8.075 |
| ast_timings | 24.149 |
| describe_scenes | 17.457 |
| summarize_scenes | 22.969 |
| synthesize_synopsis | 26.473 |
| make_embedding | 3.046 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 33.639 |
| branch_yolo_total | 10.103 |
| branch_audio_total | 46.041 |
