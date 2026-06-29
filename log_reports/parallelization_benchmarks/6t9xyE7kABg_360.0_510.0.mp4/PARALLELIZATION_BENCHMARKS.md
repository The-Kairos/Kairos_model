# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 12:58:10 UTC | 6t9xyE7kABg_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 161.985 | 0.840 | 55.833 | 13.379 | 15.962 | 13.700 | 3.895 |

## 2026-06-24 12:58:10 UTC | 6t9xyE7kABg_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/6t9xyE7kABg_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `161.985` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.840 |
| save_clips | - |
| sample_frames | 1.492 |
| caption_frames | 43.915 |
| sample_fps | 2.477 |
| detect_object_yolo | 9.102 |
| audio_scan | 13.862 |
| asr_timings | 9.159 |
| ast_timings | 32.803 |
| describe_scenes | 13.379 |
| summarize_scenes | 15.962 |
| synthesize_synopsis | 13.700 |
| make_embedding | 3.895 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.414 |
| branch_yolo_total | 11.585 |
| branch_audio_total | 55.833 |
