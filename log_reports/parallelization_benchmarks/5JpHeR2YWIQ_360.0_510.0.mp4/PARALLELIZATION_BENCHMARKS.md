# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 11:44:56 UTC | 5JpHeR2YWIQ_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 119.967 | 0.802 | 40.461 | 5.978 | 10.494 | 22.271 | 2.514 |

## 2026-06-24 11:44:56 UTC | 5JpHeR2YWIQ_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/5JpHeR2YWIQ_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `119.967` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.802 |
| save_clips | - |
| sample_frames | 0.677 |
| caption_frames | 26.314 |
| sample_fps | 2.022 |
| detect_object_yolo | 7.037 |
| audio_scan | 9.628 |
| asr_timings | 12.305 |
| ast_timings | 18.519 |
| describe_scenes | 5.978 |
| summarize_scenes | 10.494 |
| synthesize_synopsis | 22.271 |
| make_embedding | 2.514 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 26.997 |
| branch_yolo_total | 9.066 |
| branch_audio_total | 40.461 |
