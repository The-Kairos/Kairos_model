# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 02:08:33 UTC | Gk0eUHgITcw_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 142.072 | 0.641 | 45.377 | 7.272 | 22.722 | 19.797 | 3.006 |

## 2026-06-25 02:08:33 UTC | Gk0eUHgITcw_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Gk0eUHgITcw_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `142.072` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.641 |
| save_clips | - |
| sample_frames | 0.778 |
| caption_frames | 31.208 |
| sample_fps | 1.944 |
| detect_object_yolo | 7.932 |
| audio_scan | 12.849 |
| asr_timings | 8.340 |
| ast_timings | 24.180 |
| describe_scenes | 7.272 |
| summarize_scenes | 22.722 |
| synthesize_synopsis | 19.797 |
| make_embedding | 3.006 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 31.992 |
| branch_yolo_total | 9.882 |
| branch_audio_total | 45.377 |
