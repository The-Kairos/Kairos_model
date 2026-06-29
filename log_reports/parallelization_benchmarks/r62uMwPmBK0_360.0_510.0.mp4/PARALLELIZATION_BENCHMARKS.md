# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 18:21:01 UTC | r62uMwPmBK0_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 94.364 | 0.641 | 32.527 | 16.317 | 12.745 | 15.239 | 3.250 |

## 2026-06-26 18:21:01 UTC | r62uMwPmBK0_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/r62uMwPmBK0_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `94.364` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.641 |
| save_clips | - |
| sample_frames | 1.017 |
| caption_frames | 31.504 |
| sample_fps | 2.104 |
| detect_object_yolo | 8.993 |
| audio_scan | 1.079 |
| asr_timings | 0.000 |
| ast_timings | 0.000 |
| describe_scenes | 16.317 |
| summarize_scenes | 12.745 |
| synthesize_synopsis | 15.239 |
| make_embedding | 3.250 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.527 |
| branch_yolo_total | 11.102 |
| branch_audio_total | 1.086 |
