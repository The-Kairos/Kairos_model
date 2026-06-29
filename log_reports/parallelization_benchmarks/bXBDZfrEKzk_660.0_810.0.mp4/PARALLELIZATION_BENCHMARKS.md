# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 01:29:08 UTC | bXBDZfrEKzk_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 112.476 | 0.773 | 43.535 | 8.497 | 6.628 | 13.506 | 2.515 |

## 2026-06-26 01:29:08 UTC | bXBDZfrEKzk_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/bXBDZfrEKzk_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `112.476` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.773 |
| save_clips | - |
| sample_frames | 0.629 |
| caption_frames | 25.114 |
| sample_fps | 2.024 |
| detect_object_yolo | 7.862 |
| audio_scan | 16.130 |
| asr_timings | 8.519 |
| ast_timings | 18.877 |
| describe_scenes | 8.497 |
| summarize_scenes | 6.628 |
| synthesize_synopsis | 13.506 |
| make_embedding | 2.515 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 25.749 |
| branch_yolo_total | 9.892 |
| branch_audio_total | 43.535 |
