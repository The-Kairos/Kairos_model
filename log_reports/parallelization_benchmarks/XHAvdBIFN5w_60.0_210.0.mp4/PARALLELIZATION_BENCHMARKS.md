# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 21:31:16 UTC | XHAvdBIFN5w_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 149.173 | 0.619 | 54.595 | 13.813 | 13.931 | 9.323 | 3.544 |

## 2026-06-25 21:31:16 UTC | XHAvdBIFN5w_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/XHAvdBIFN5w_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `149.173` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.619 |
| save_clips | - |
| sample_frames | 1.015 |
| caption_frames | 40.432 |
| sample_fps | 1.976 |
| detect_object_yolo | 8.509 |
| audio_scan | 15.720 |
| asr_timings | 9.360 |
| ast_timings | 29.507 |
| describe_scenes | 13.813 |
| summarize_scenes | 13.931 |
| synthesize_synopsis | 9.323 |
| make_embedding | 3.544 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.453 |
| branch_yolo_total | 10.491 |
| branch_audio_total | 54.595 |
