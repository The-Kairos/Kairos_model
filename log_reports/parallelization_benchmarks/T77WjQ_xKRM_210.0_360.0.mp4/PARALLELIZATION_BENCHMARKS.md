# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 17:41:18 UTC | T77WjQ_xKRM_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 168.518 | 0.661 | 61.101 | 8.766 | 30.483 | 16.920 | 3.023 |

## 2026-06-25 17:41:18 UTC | T77WjQ_xKRM_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/T77WjQ_xKRM_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `168.518` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.661 |
| save_clips | - |
| sample_frames | 0.977 |
| caption_frames | 34.512 |
| sample_fps | 2.091 |
| detect_object_yolo | 8.576 |
| audio_scan | 11.753 |
| asr_timings | 24.630 |
| ast_timings | 24.710 |
| describe_scenes | 8.766 |
| summarize_scenes | 30.483 |
| synthesize_synopsis | 16.920 |
| make_embedding | 3.023 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 35.494 |
| branch_yolo_total | 10.673 |
| branch_audio_total | 61.101 |
