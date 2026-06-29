# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 06:14:31 UTC | zpRzgcjq0x8_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 194.035 | 0.760 | 70.094 | 17.635 | 9.893 | 7.366 | 6.157 |

## 2026-06-27 06:14:31 UTC | zpRzgcjq0x8_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/zpRzgcjq0x8_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `194.035` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.760 |
| save_clips | - |
| sample_frames | 1.836 |
| caption_frames | 64.263 |
| sample_fps | 2.580 |
| detect_object_yolo | 12.045 |
| audio_scan | 12.662 |
| asr_timings | 7.453 |
| ast_timings | 49.972 |
| describe_scenes | 17.635 |
| summarize_scenes | 9.893 |
| synthesize_synopsis | 7.366 |
| make_embedding | 6.157 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 66.105 |
| branch_yolo_total | 14.632 |
| branch_audio_total | 70.094 |
