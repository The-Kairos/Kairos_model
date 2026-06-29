# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 18:59:01 UTC | ro2DUWXm_i4_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 180.122 | 0.786 | 58.725 | 23.363 | 11.561 | 16.277 | 4.507 |

## 2026-06-26 18:59:01 UTC | ro2DUWXm_i4_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ro2DUWXm_i4_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `180.122` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.786 |
| save_clips | - |
| sample_frames | 1.532 |
| caption_frames | 49.333 |
| sample_fps | 2.381 |
| detect_object_yolo | 10.217 |
| audio_scan | 11.759 |
| asr_timings | 8.816 |
| ast_timings | 38.141 |
| describe_scenes | 23.363 |
| summarize_scenes | 11.561 |
| synthesize_synopsis | 16.277 |
| make_embedding | 4.507 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 50.871 |
| branch_yolo_total | 12.605 |
| branch_audio_total | 58.725 |
