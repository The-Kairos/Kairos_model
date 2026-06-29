# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 05:57:50 UTC | zeQ4qrFwePM_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 151.698 | 0.798 | 58.336 | 11.624 | 7.217 | 8.881 | 4.103 |

## 2026-06-27 05:57:50 UTC | zeQ4qrFwePM_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/zeQ4qrFwePM_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `151.698` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.798 |
| save_clips | - |
| sample_frames | 1.386 |
| caption_frames | 45.877 |
| sample_fps | 2.441 |
| detect_object_yolo | 9.647 |
| audio_scan | 12.872 |
| asr_timings | 9.838 |
| ast_timings | 35.617 |
| describe_scenes | 11.624 |
| summarize_scenes | 7.217 |
| synthesize_synopsis | 8.881 |
| make_embedding | 4.103 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.269 |
| branch_yolo_total | 12.094 |
| branch_audio_total | 58.336 |
