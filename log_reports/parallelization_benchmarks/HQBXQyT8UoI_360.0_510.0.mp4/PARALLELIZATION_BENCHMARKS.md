# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 03:32:46 UTC | HQBXQyT8UoI_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 189.948 | 0.814 | 62.750 | 17.601 | 27.226 | 13.734 | 4.209 |

## 2026-06-25 03:32:46 UTC | HQBXQyT8UoI_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/HQBXQyT8UoI_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `189.948` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.814 |
| save_clips | - |
| sample_frames | 1.419 |
| caption_frames | 47.852 |
| sample_fps | 2.487 |
| detect_object_yolo | 10.400 |
| audio_scan | 12.891 |
| asr_timings | 14.437 |
| ast_timings | 35.414 |
| describe_scenes | 17.601 |
| summarize_scenes | 27.226 |
| synthesize_synopsis | 13.734 |
| make_embedding | 4.209 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.277 |
| branch_yolo_total | 12.893 |
| branch_audio_total | 62.750 |
