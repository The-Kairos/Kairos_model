# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 05:02:29 UTC | J7N2j6leva4_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 104.505 | 0.655 | 37.953 | 8.635 | 10.197 | 22.122 | 1.835 |

## 2026-06-25 05:02:29 UTC | J7N2j6leva4_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/J7N2j6leva4_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `104.505` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.655 |
| save_clips | - |
| sample_frames | 0.351 |
| caption_frames | 13.515 |
| sample_fps | 1.716 |
| detect_object_yolo | 6.142 |
| audio_scan | 16.011 |
| asr_timings | 12.077 |
| ast_timings | 9.856 |
| describe_scenes | 8.635 |
| summarize_scenes | 10.197 |
| synthesize_synopsis | 22.122 |
| make_embedding | 1.835 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 13.871 |
| branch_yolo_total | 7.864 |
| branch_audio_total | 37.953 |
