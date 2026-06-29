# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 14:12:26 UTC | kmJG4kjQXL8_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 226.662 | 0.796 | 67.395 | 37.808 | 15.846 | 24.677 | 5.660 |

## 2026-06-26 14:12:26 UTC | kmJG4kjQXL8_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/kmJG4kjQXL8_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `226.662` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.796 |
| save_clips | - |
| sample_frames | 1.512 |
| caption_frames | 57.570 |
| sample_fps | 2.535 |
| detect_object_yolo | 11.412 |
| audio_scan | 14.069 |
| asr_timings | 8.881 |
| ast_timings | 44.437 |
| describe_scenes | 37.808 |
| summarize_scenes | 15.846 |
| synthesize_synopsis | 24.677 |
| make_embedding | 5.660 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 59.088 |
| branch_yolo_total | 13.953 |
| branch_audio_total | 67.395 |
