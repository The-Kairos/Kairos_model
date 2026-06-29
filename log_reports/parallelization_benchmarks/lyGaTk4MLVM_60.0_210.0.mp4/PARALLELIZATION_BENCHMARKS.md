# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 17:44:12 UTC | lyGaTk4MLVM_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 108.345 | 0.716 | 35.158 | 8.197 | 17.140 | 14.098 | 4.251 |

## 2026-06-26 17:44:12 UTC | lyGaTk4MLVM_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/lyGaTk4MLVM_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `108.345` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.716 |
| save_clips | - |
| sample_frames | 0.347 |
| caption_frames | 18.996 |
| sample_fps | 1.800 |
| detect_object_yolo | 6.242 |
| audio_scan | 13.977 |
| asr_timings | 8.035 |
| ast_timings | 13.136 |
| describe_scenes | 8.197 |
| summarize_scenes | 17.140 |
| synthesize_synopsis | 14.098 |
| make_embedding | 4.251 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 19.348 |
| branch_yolo_total | 8.048 |
| branch_audio_total | 35.158 |
