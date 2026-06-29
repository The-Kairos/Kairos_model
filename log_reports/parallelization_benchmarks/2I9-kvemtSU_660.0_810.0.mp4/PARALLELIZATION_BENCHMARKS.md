# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 21:15:02 UTC | 2I9-kvemtSU_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 164.329 | 0.770 | 59.408 | 19.522 | 7.366 | 8.011 | 4.496 |

## 2026-06-21 21:15:02 UTC | 2I9-kvemtSU_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2I9-kvemtSU_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `164.329` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.770 |
| save_clips | - |
| sample_frames | 1.407 |
| caption_frames | 49.178 |
| sample_fps | 2.416 |
| detect_object_yolo | 10.363 |
| audio_scan | 14.872 |
| asr_timings | 7.693 |
| ast_timings | 36.835 |
| describe_scenes | 19.522 |
| summarize_scenes | 7.366 |
| synthesize_synopsis | 8.011 |
| make_embedding | 4.496 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 50.591 |
| branch_yolo_total | 12.784 |
| branch_audio_total | 59.408 |
