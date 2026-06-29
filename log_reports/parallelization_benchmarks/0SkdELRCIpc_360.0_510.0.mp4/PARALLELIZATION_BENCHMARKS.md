# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 13:33:18 UTC | 0SkdELRCIpc_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 96.865 | 0.786 | 37.797 | 7.476 | 5.584 | 9.512 | 2.331 |

## 2026-06-27 13:33:18 UTC | 0SkdELRCIpc_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0SkdELRCIpc_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `96.865` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.786 |
| save_clips | - |
| sample_frames | 0.581 |
| caption_frames | 22.653 |
| sample_fps | 2.033 |
| detect_object_yolo | 6.617 |
| audio_scan | 12.919 |
| asr_timings | 9.015 |
| ast_timings | 15.854 |
| describe_scenes | 7.476 |
| summarize_scenes | 5.584 |
| synthesize_synopsis | 9.512 |
| make_embedding | 2.331 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 23.240 |
| branch_yolo_total | 8.656 |
| branch_audio_total | 37.797 |
