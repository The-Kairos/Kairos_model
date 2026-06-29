# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 00:56:06 UTC | b5DIp2wHa8Q_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 138.381 | 0.644 | 50.369 | 15.533 | 9.638 | 12.023 | 3.088 |

## 2026-06-26 00:56:06 UTC | b5DIp2wHa8Q_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/b5DIp2wHa8Q_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `138.381` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.644 |
| save_clips | - |
| sample_frames | 0.769 |
| caption_frames | 34.541 |
| sample_fps | 1.960 |
| detect_object_yolo | 8.327 |
| audio_scan | 13.630 |
| asr_timings | 12.088 |
| ast_timings | 24.642 |
| describe_scenes | 15.533 |
| summarize_scenes | 9.638 |
| synthesize_synopsis | 12.023 |
| make_embedding | 3.088 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 35.315 |
| branch_yolo_total | 10.293 |
| branch_audio_total | 50.369 |
