# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 05:55:17 UTC | zeQ4qrFwePM_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 130.935 | 0.811 | 48.371 | 9.410 | 6.560 | 7.183 | 3.575 |

## 2026-06-27 05:55:17 UTC | zeQ4qrFwePM_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/zeQ4qrFwePM_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `130.935` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.811 |
| save_clips | - |
| sample_frames | 1.230 |
| caption_frames | 40.720 |
| sample_fps | 2.366 |
| detect_object_yolo | 9.242 |
| audio_scan | 7.624 |
| asr_timings | 10.690 |
| ast_timings | 30.048 |
| describe_scenes | 9.410 |
| summarize_scenes | 6.560 |
| synthesize_synopsis | 7.183 |
| make_embedding | 3.575 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.956 |
| branch_yolo_total | 11.614 |
| branch_audio_total | 48.371 |
