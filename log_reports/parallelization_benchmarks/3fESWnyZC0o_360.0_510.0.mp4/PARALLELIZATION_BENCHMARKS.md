# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 22:16:39 UTC | 3fESWnyZC0o_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 164.846 | 0.758 | 61.183 | 15.622 | 8.698 | 7.673 | 5.038 |

## 2026-06-21 22:16:39 UTC | 3fESWnyZC0o_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3fESWnyZC0o_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `164.846` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.758 |
| save_clips | - |
| sample_frames | 1.378 |
| caption_frames | 50.304 |
| sample_fps | 2.391 |
| detect_object_yolo | 10.360 |
| audio_scan | 15.977 |
| asr_timings | 7.616 |
| ast_timings | 37.582 |
| describe_scenes | 15.622 |
| summarize_scenes | 8.698 |
| synthesize_synopsis | 7.673 |
| make_embedding | 5.038 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.688 |
| branch_yolo_total | 12.757 |
| branch_audio_total | 61.183 |
