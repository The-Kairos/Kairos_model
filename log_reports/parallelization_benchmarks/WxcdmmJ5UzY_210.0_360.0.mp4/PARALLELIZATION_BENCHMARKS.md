# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 21:18:07 UTC | WxcdmmJ5UzY_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 102.215 | 0.692 | 41.488 | 7.429 | 4.718 | 13.941 | 2.574 |

## 2026-06-25 21:18:07 UTC | WxcdmmJ5UzY_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/WxcdmmJ5UzY_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `102.215` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.692 |
| save_clips | - |
| sample_frames | 1.016 |
| caption_frames | 20.918 |
| sample_fps | 1.872 |
| detect_object_yolo | 6.158 |
| audio_scan | 14.301 |
| asr_timings | 8.133 |
| ast_timings | 19.045 |
| describe_scenes | 7.429 |
| summarize_scenes | 4.718 |
| synthesize_synopsis | 13.941 |
| make_embedding | 2.574 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 21.940 |
| branch_yolo_total | 8.035 |
| branch_audio_total | 41.488 |
