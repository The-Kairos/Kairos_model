# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 06:28:37 UTC | hfJvu-roZGQ_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 148.997 | 0.622 | 81.321 | 4.332 | 9.269 | 21.927 | 2.049 |

## 2026-06-26 06:28:37 UTC | hfJvu-roZGQ_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/hfJvu-roZGQ_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `148.997` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.622 |
| save_clips | - |
| sample_frames | 0.349 |
| caption_frames | 19.458 |
| sample_fps | 0.705 |
| detect_object_yolo | 7.556 |
| audio_scan | 15.068 |
| asr_timings | 53.935 |
| ast_timings | 12.309 |
| describe_scenes | 4.332 |
| summarize_scenes | 9.269 |
| synthesize_synopsis | 21.927 |
| make_embedding | 2.049 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 19.814 |
| branch_yolo_total | 8.267 |
| branch_audio_total | 81.321 |
