# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 03:20:46 UTC | wfIWTD30gCw_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 118.764 | 0.780 | 48.922 | 6.865 | 8.429 | 7.826 | 2.749 |

## 2026-06-27 03:20:46 UTC | wfIWTD30gCw_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/wfIWTD30gCw_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `118.764` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.780 |
| save_clips | - |
| sample_frames | 0.663 |
| caption_frames | 31.178 |
| sample_fps | 2.094 |
| detect_object_yolo | 7.839 |
| audio_scan | 10.891 |
| asr_timings | 16.116 |
| ast_timings | 21.906 |
| describe_scenes | 6.865 |
| summarize_scenes | 8.429 |
| synthesize_synopsis | 7.826 |
| make_embedding | 2.749 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 31.846 |
| branch_yolo_total | 9.939 |
| branch_audio_total | 48.922 |
