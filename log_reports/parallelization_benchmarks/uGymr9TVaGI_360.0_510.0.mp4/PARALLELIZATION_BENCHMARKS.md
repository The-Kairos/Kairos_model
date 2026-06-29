# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 00:49:53 UTC | uGymr9TVaGI_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 68.839 | 0.780 | 31.268 | 5.736 | 3.193 | 6.693 | 1.485 |

## 2026-06-27 00:49:53 UTC | uGymr9TVaGI_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/uGymr9TVaGI_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `68.839` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.780 |
| save_clips | - |
| sample_frames | 0.167 |
| caption_frames | 10.647 |
| sample_fps | 1.805 |
| detect_object_yolo | 5.593 |
| audio_scan | 14.970 |
| asr_timings | 8.211 |
| ast_timings | 8.077 |
| describe_scenes | 5.736 |
| summarize_scenes | 3.193 |
| synthesize_synopsis | 6.693 |
| make_embedding | 1.485 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 10.820 |
| branch_yolo_total | 7.404 |
| branch_audio_total | 31.268 |
