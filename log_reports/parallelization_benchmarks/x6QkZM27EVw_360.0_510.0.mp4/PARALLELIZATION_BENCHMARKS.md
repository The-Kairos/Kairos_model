# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 03:27:56 UTC | x6QkZM27EVw_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 168.511 | 0.645 | 89.180 | 10.756 | 6.256 | 6.596 | 3.247 |

## 2026-06-27 03:27:56 UTC | x6QkZM27EVw_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/x6QkZM27EVw_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `168.511` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.645 |
| save_clips | - |
| sample_frames | 1.174 |
| caption_frames | 38.245 |
| sample_fps | 2.165 |
| detect_object_yolo | 8.846 |
| audio_scan | 14.062 |
| asr_timings | 48.183 |
| ast_timings | 26.927 |
| describe_scenes | 10.756 |
| summarize_scenes | 6.256 |
| synthesize_synopsis | 6.596 |
| make_embedding | 3.247 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.425 |
| branch_yolo_total | 11.017 |
| branch_audio_total | 89.180 |
