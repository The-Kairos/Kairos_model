# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 09:09:51 UTC | iaWkqZHf_qk_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 88.533 | 0.353 | 29.951 | 5.380 | 6.563 | 23.987 | 1.577 |

## 2026-06-26 09:09:51 UTC | iaWkqZHf_qk_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/iaWkqZHf_qk_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `88.533` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.353 |
| save_clips | - |
| sample_frames | 0.248 |
| caption_frames | 12.239 |
| sample_fps | 1.343 |
| detect_object_yolo | 5.406 |
| audio_scan | 11.995 |
| asr_timings | 10.664 |
| ast_timings | 7.283 |
| describe_scenes | 5.380 |
| summarize_scenes | 6.563 |
| synthesize_synopsis | 23.987 |
| make_embedding | 1.577 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 12.493 |
| branch_yolo_total | 6.755 |
| branch_audio_total | 29.951 |
