# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 10:39:01 UTC | N4VtpYgZLVg_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 236.309 | 0.637 | 65.053 | 26.390 | 38.781 | 33.455 | 4.501 |

## 2026-06-25 10:39:01 UTC | N4VtpYgZLVg_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/N4VtpYgZLVg_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `236.309` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.637 |
| save_clips | - |
| sample_frames | 1.305 |
| caption_frames | 52.136 |
| sample_fps | 2.245 |
| detect_object_yolo | 10.359 |
| audio_scan | 15.998 |
| asr_timings | 11.138 |
| ast_timings | 37.909 |
| describe_scenes | 26.390 |
| summarize_scenes | 38.781 |
| synthesize_synopsis | 33.455 |
| make_embedding | 4.501 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 53.447 |
| branch_yolo_total | 12.610 |
| branch_audio_total | 65.053 |
