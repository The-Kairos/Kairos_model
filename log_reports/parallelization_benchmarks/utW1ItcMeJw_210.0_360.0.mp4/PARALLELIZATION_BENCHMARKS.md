# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 01:41:46 UTC | utW1ItcMeJw_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 184.632 | 0.790 | 71.115 | 15.290 | 8.819 | 7.251 | 5.367 |

## 2026-06-27 01:41:46 UTC | utW1ItcMeJw_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/utW1ItcMeJw_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `184.632` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.790 |
| save_clips | - |
| sample_frames | 1.341 |
| caption_frames | 58.847 |
| sample_fps | 2.515 |
| detect_object_yolo | 11.871 |
| audio_scan | 15.046 |
| asr_timings | 12.491 |
| ast_timings | 43.569 |
| describe_scenes | 15.290 |
| summarize_scenes | 8.819 |
| synthesize_synopsis | 7.251 |
| make_embedding | 5.367 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 60.193 |
| branch_yolo_total | 14.392 |
| branch_audio_total | 71.115 |
