# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 08:08:50 UTC | iCbjk3uRtqA_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 214.618 | 0.782 | 59.710 | 18.136 | 38.974 | 41.282 | 3.620 |

## 2026-06-26 08:08:50 UTC | iCbjk3uRtqA_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/iCbjk3uRtqA_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `214.618` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.782 |
| save_clips | - |
| sample_frames | 0.947 |
| caption_frames | 38.690 |
| sample_fps | 2.176 |
| detect_object_yolo | 8.893 |
| audio_scan | 15.148 |
| asr_timings | 14.675 |
| ast_timings | 29.879 |
| describe_scenes | 18.136 |
| summarize_scenes | 38.974 |
| synthesize_synopsis | 41.282 |
| make_embedding | 3.620 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.643 |
| branch_yolo_total | 11.076 |
| branch_audio_total | 59.710 |
