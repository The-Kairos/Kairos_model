# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 13:51:47 UTC | 0iz-ty5Wl3U_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 127.805 | 0.669 | 51.009 | 8.400 | 6.054 | 8.722 | 3.302 |

## 2026-06-27 13:51:47 UTC | 0iz-ty5Wl3U_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0iz-ty5Wl3U_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `127.805` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.669 |
| save_clips | - |
| sample_frames | 0.898 |
| caption_frames | 36.905 |
| sample_fps | 2.068 |
| detect_object_yolo | 8.362 |
| audio_scan | 14.961 |
| asr_timings | 9.767 |
| ast_timings | 26.273 |
| describe_scenes | 8.400 |
| summarize_scenes | 6.054 |
| synthesize_synopsis | 8.722 |
| make_embedding | 3.302 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.809 |
| branch_yolo_total | 10.436 |
| branch_audio_total | 51.009 |
