# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 12:02:36 UTC | jqfCtEdMIfY_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 204.286 | 0.807 | 100.141 | 20.903 | 16.320 | 14.546 | 3.242 |

## 2026-06-26 12:02:36 UTC | jqfCtEdMIfY_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jqfCtEdMIfY_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `204.286` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.807 |
| save_clips | - |
| sample_frames | 1.079 |
| caption_frames | 35.010 |
| sample_fps | 2.189 |
| detect_object_yolo | 8.626 |
| audio_scan | 14.048 |
| asr_timings | 60.038 |
| ast_timings | 26.046 |
| describe_scenes | 20.903 |
| summarize_scenes | 16.320 |
| synthesize_synopsis | 14.546 |
| make_embedding | 3.242 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.095 |
| branch_yolo_total | 10.820 |
| branch_audio_total | 100.141 |
