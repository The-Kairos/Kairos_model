# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 00:50:59 UTC | b5DIp2wHa8Q_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 151.457 | 0.641 | 58.913 | 12.968 | 16.636 | 9.347 | 3.287 |

## 2026-06-26 00:50:59 UTC | b5DIp2wHa8Q_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/b5DIp2wHa8Q_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `151.457` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.641 |
| save_clips | - |
| sample_frames | 0.878 |
| caption_frames | 36.757 |
| sample_fps | 2.003 |
| detect_object_yolo | 8.607 |
| audio_scan | 12.748 |
| asr_timings | 18.947 |
| ast_timings | 27.209 |
| describe_scenes | 12.968 |
| summarize_scenes | 16.636 |
| synthesize_synopsis | 9.347 |
| make_embedding | 3.287 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.640 |
| branch_yolo_total | 10.616 |
| branch_audio_total | 58.913 |
