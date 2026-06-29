# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 15:33:15 UTC | QrFLjLZIeig_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 159.772 | 0.809 | 54.965 | 16.797 | 13.965 | 23.754 | 3.177 |

## 2026-06-25 15:33:15 UTC | QrFLjLZIeig_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/QrFLjLZIeig_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `159.772` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.809 |
| save_clips | - |
| sample_frames | 0.981 |
| caption_frames | 33.762 |
| sample_fps | 2.195 |
| detect_object_yolo | 7.951 |
| audio_scan | 16.694 |
| asr_timings | 13.607 |
| ast_timings | 24.655 |
| describe_scenes | 16.797 |
| summarize_scenes | 13.965 |
| synthesize_synopsis | 23.754 |
| make_embedding | 3.177 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 34.749 |
| branch_yolo_total | 10.152 |
| branch_audio_total | 54.965 |
