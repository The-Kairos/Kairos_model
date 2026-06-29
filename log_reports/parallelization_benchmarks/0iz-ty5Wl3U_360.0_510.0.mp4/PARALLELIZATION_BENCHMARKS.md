# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 13:53:38 UTC | 0iz-ty5Wl3U_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 110.052 | 0.665 | 41.708 | 6.850 | 7.114 | 10.242 | 2.788 |

## 2026-06-27 13:53:38 UTC | 0iz-ty5Wl3U_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0iz-ty5Wl3U_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `110.052` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.665 |
| save_clips | - |
| sample_frames | 0.702 |
| caption_frames | 28.735 |
| sample_fps | 1.946 |
| detect_object_yolo | 7.917 |
| audio_scan | 12.754 |
| asr_timings | 8.483 |
| ast_timings | 20.463 |
| describe_scenes | 6.850 |
| summarize_scenes | 7.114 |
| synthesize_synopsis | 10.242 |
| make_embedding | 2.788 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 29.444 |
| branch_yolo_total | 9.868 |
| branch_audio_total | 41.708 |
