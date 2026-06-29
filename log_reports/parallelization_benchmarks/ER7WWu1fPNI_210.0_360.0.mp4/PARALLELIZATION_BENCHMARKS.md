# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 23:35:37 UTC | ER7WWu1fPNI_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 121.589 | 0.777 | 39.258 | 12.375 | 9.305 | 9.801 | 3.603 |

## 2026-06-24 23:35:37 UTC | ER7WWu1fPNI_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ER7WWu1fPNI_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `121.589` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.777 |
| save_clips | - |
| sample_frames | 1.432 |
| caption_frames | 37.820 |
| sample_fps | 2.325 |
| detect_object_yolo | 8.771 |
| audio_scan | 3.744 |
| asr_timings | 0.000 |
| ast_timings | 30.203 |
| describe_scenes | 12.375 |
| summarize_scenes | 9.305 |
| synthesize_synopsis | 9.801 |
| make_embedding | 3.603 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.258 |
| branch_yolo_total | 11.102 |
| branch_audio_total | 33.956 |
