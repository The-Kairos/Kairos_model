# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 18:41:32 UTC | U3QUfxqnfJg_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 158.947 | 0.657 | 57.831 | 14.188 | 8.024 | 23.681 | 3.298 |

## 2026-06-25 18:41:32 UTC | U3QUfxqnfJg_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/U3QUfxqnfJg_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `158.947` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.657 |
| save_clips | - |
| sample_frames | 0.841 |
| caption_frames | 38.461 |
| sample_fps | 2.008 |
| detect_object_yolo | 8.551 |
| audio_scan | 10.755 |
| asr_timings | 19.598 |
| ast_timings | 27.470 |
| describe_scenes | 14.188 |
| summarize_scenes | 8.024 |
| synthesize_synopsis | 23.681 |
| make_embedding | 3.298 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.308 |
| branch_yolo_total | 10.564 |
| branch_audio_total | 57.831 |
