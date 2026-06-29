# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 04:17:04 UTC | fMStD2X1434_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 163.336 | 0.813 | 73.380 | 11.622 | 8.593 | 11.287 | 3.609 |

## 2026-06-26 04:17:04 UTC | fMStD2X1434_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/fMStD2X1434_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `163.336` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.813 |
| save_clips | - |
| sample_frames | 1.216 |
| caption_frames | 39.881 |
| sample_fps | 2.355 |
| detect_object_yolo | 9.161 |
| audio_scan | 15.236 |
| asr_timings | 27.743 |
| ast_timings | 30.393 |
| describe_scenes | 11.622 |
| summarize_scenes | 8.593 |
| synthesize_synopsis | 11.287 |
| make_embedding | 3.609 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.102 |
| branch_yolo_total | 11.522 |
| branch_audio_total | 73.380 |
