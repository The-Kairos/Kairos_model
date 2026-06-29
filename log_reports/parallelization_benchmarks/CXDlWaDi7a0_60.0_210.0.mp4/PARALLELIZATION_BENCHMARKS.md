# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 20:29:33 UTC | CXDlWaDi7a0_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 172.444 | 0.625 | 59.949 | 12.279 | 20.060 | 16.363 | 3.793 |

## 2026-06-24 20:29:33 UTC | CXDlWaDi7a0_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/CXDlWaDi7a0_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `172.444` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.625 |
| save_clips | - |
| sample_frames | 1.034 |
| caption_frames | 45.541 |
| sample_fps | 2.103 |
| detect_object_yolo | 9.312 |
| audio_scan | 16.039 |
| asr_timings | 11.247 |
| ast_timings | 32.654 |
| describe_scenes | 12.279 |
| summarize_scenes | 20.060 |
| synthesize_synopsis | 16.363 |
| make_embedding | 3.793 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.582 |
| branch_yolo_total | 11.421 |
| branch_audio_total | 59.949 |
