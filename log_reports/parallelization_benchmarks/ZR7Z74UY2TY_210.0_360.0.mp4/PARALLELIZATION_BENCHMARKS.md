# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 22:23:46 UTC | ZR7Z74UY2TY_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 139.732 | 0.678 | 54.360 | 9.211 | 11.814 | 15.079 | 3.014 |

## 2026-06-25 22:23:46 UTC | ZR7Z74UY2TY_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ZR7Z74UY2TY_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `139.732` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.678 |
| save_clips | - |
| sample_frames | 1.065 |
| caption_frames | 33.215 |
| sample_fps | 2.094 |
| detect_object_yolo | 7.794 |
| audio_scan | 12.863 |
| asr_timings | 16.203 |
| ast_timings | 25.285 |
| describe_scenes | 9.211 |
| summarize_scenes | 11.814 |
| synthesize_synopsis | 15.079 |
| make_embedding | 3.014 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 34.286 |
| branch_yolo_total | 9.894 |
| branch_audio_total | 54.360 |
