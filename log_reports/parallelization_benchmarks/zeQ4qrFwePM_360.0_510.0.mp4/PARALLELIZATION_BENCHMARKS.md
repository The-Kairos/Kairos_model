# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 05:53:05 UTC | zeQ4qrFwePM_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 158.328 | 0.782 | 57.316 | 10.137 | 13.662 | 9.758 | 4.168 |

## 2026-06-27 05:53:05 UTC | zeQ4qrFwePM_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/zeQ4qrFwePM_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `158.328` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.782 |
| save_clips | - |
| sample_frames | 1.137 |
| caption_frames | 47.832 |
| sample_fps | 2.340 |
| detect_object_yolo | 9.760 |
| audio_scan | 12.820 |
| asr_timings | 9.105 |
| ast_timings | 35.382 |
| describe_scenes | 10.137 |
| summarize_scenes | 13.662 |
| synthesize_synopsis | 9.758 |
| make_embedding | 4.168 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 48.975 |
| branch_yolo_total | 12.106 |
| branch_audio_total | 57.316 |
