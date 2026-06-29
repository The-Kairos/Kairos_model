# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 13:59:37 UTC | 0iz-ty5Wl3U_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 110.437 | 0.637 | 45.443 | 7.248 | 6.276 | 8.567 | 2.728 |

## 2026-06-27 13:59:37 UTC | 0iz-ty5Wl3U_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0iz-ty5Wl3U_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `110.437` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.637 |
| save_clips | - |
| sample_frames | 0.750 |
| caption_frames | 28.106 |
| sample_fps | 1.936 |
| detect_object_yolo | 7.350 |
| audio_scan | 15.944 |
| asr_timings | 8.751 |
| ast_timings | 20.739 |
| describe_scenes | 7.248 |
| summarize_scenes | 6.276 |
| synthesize_synopsis | 8.567 |
| make_embedding | 2.728 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 28.862 |
| branch_yolo_total | 9.292 |
| branch_audio_total | 45.443 |
