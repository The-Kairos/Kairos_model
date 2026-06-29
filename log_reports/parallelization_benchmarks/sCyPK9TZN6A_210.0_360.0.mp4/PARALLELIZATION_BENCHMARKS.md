# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 19:40:26 UTC | sCyPK9TZN6A_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 78.449 | 0.782 | 28.560 | 6.619 | 3.369 | 15.997 | 1.575 |

## 2026-06-26 19:40:26 UTC | sCyPK9TZN6A_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/sCyPK9TZN6A_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `78.449` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.782 |
| save_clips | - |
| sample_frames | 0.224 |
| caption_frames | 11.895 |
| sample_fps | 1.808 |
| detect_object_yolo | 6.168 |
| audio_scan | 9.717 |
| asr_timings | 11.552 |
| ast_timings | 7.282 |
| describe_scenes | 6.619 |
| summarize_scenes | 3.369 |
| synthesize_synopsis | 15.997 |
| make_embedding | 1.575 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 12.125 |
| branch_yolo_total | 7.983 |
| branch_audio_total | 28.560 |
