# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 14:17:32 UTC | kr_sTfYX_FI_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 173.375 | 0.792 | 72.705 | 12.014 | 23.306 | 17.578 | 3.027 |

## 2026-06-26 14:17:32 UTC | kr_sTfYX_FI_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/kr_sTfYX_FI_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `173.375` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.792 |
| save_clips | - |
| sample_frames | 0.811 |
| caption_frames | 31.607 |
| sample_fps | 2.140 |
| detect_object_yolo | 7.966 |
| audio_scan | 14.072 |
| asr_timings | 34.089 |
| ast_timings | 24.535 |
| describe_scenes | 12.014 |
| summarize_scenes | 23.306 |
| synthesize_synopsis | 17.578 |
| make_embedding | 3.027 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.424 |
| branch_yolo_total | 10.112 |
| branch_audio_total | 72.705 |
