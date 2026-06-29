# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 22:59:01 UTC | srP7RFVdjWc_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 307.810 | 0.839 | 109.822 | 33.011 | 19.391 | 9.931 | 7.537 |

## 2026-06-26 22:59:01 UTC | srP7RFVdjWc_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/srP7RFVdjWc_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `307.810` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.839 |
| save_clips | - |
| sample_frames | 2.772 |
| caption_frames | 107.044 |
| sample_fps | 3.341 |
| detect_object_yolo | 17.950 |
| audio_scan | 14.946 |
| asr_timings | 8.734 |
| ast_timings | 80.862 |
| describe_scenes | 33.011 |
| summarize_scenes | 19.391 |
| synthesize_synopsis | 9.931 |
| make_embedding | 7.537 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 109.822 |
| branch_yolo_total | 21.297 |
| branch_audio_total | 104.550 |
