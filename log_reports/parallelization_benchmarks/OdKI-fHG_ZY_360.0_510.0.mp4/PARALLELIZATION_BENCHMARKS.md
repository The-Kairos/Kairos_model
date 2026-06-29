# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 12:06:51 UTC | OdKI-fHG_ZY_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 217.765 | 0.703 | 63.458 | 28.946 | 20.200 | 34.439 | 4.454 |

## 2026-06-25 12:06:51 UTC | OdKI-fHG_ZY_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/OdKI-fHG_ZY_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `217.765` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.703 |
| save_clips | - |
| sample_frames | 1.407 |
| caption_frames | 50.205 |
| sample_fps | 2.344 |
| detect_object_yolo | 10.184 |
| audio_scan | 14.269 |
| asr_timings | 10.087 |
| ast_timings | 39.094 |
| describe_scenes | 28.946 |
| summarize_scenes | 20.200 |
| synthesize_synopsis | 34.439 |
| make_embedding | 4.454 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.617 |
| branch_yolo_total | 12.534 |
| branch_audio_total | 63.458 |
