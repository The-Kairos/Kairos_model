# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 13:05:42 UTC | 6v2jxLQazHc_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 150.906 | 0.664 | 45.585 | 18.604 | 13.340 | 22.914 | 3.141 |

## 2026-06-24 13:05:42 UTC | 6v2jxLQazHc_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/6v2jxLQazHc_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `150.906` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.664 |
| save_clips | - |
| sample_frames | 0.775 |
| caption_frames | 33.972 |
| sample_fps | 2.028 |
| detect_object_yolo | 8.418 |
| audio_scan | 8.637 |
| asr_timings | 12.411 |
| ast_timings | 24.528 |
| describe_scenes | 18.604 |
| summarize_scenes | 13.340 |
| synthesize_synopsis | 22.914 |
| make_embedding | 3.141 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 34.753 |
| branch_yolo_total | 10.451 |
| branch_audio_total | 45.585 |
