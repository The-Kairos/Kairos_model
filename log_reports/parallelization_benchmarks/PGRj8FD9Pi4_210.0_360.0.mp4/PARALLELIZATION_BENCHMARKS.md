# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 12:57:21 UTC | PGRj8FD9Pi4_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 208.996 | 0.627 | 62.525 | 33.931 | 18.743 | 20.265 | 5.126 |

## 2026-06-25 12:57:21 UTC | PGRj8FD9Pi4_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/PGRj8FD9Pi4_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `208.996` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.627 |
| save_clips | - |
| sample_frames | 1.214 |
| caption_frames | 52.619 |
| sample_fps | 2.253 |
| detect_object_yolo | 10.287 |
| audio_scan | 7.724 |
| asr_timings | 13.782 |
| ast_timings | 41.010 |
| describe_scenes | 33.931 |
| summarize_scenes | 18.743 |
| synthesize_synopsis | 20.265 |
| make_embedding | 5.126 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 53.839 |
| branch_yolo_total | 12.546 |
| branch_audio_total | 62.525 |
