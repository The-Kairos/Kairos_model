# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 10:49:30 UTC | jL7v3S2mFww_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 108.408 | 0.661 | 31.402 | 7.678 | 10.997 | 25.801 | 2.065 |

## 2026-06-26 10:49:30 UTC | jL7v3S2mFww_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jL7v3S2mFww_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `108.408` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.661 |
| save_clips | - |
| sample_frames | 0.398 |
| caption_frames | 19.129 |
| sample_fps | 1.819 |
| detect_object_yolo | 7.037 |
| audio_scan | 6.537 |
| asr_timings | 11.776 |
| ast_timings | 13.081 |
| describe_scenes | 7.678 |
| summarize_scenes | 10.997 |
| synthesize_synopsis | 25.801 |
| make_embedding | 2.065 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 19.532 |
| branch_yolo_total | 8.862 |
| branch_audio_total | 31.402 |
