# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 12:33:41 UTC | OkKEFnPQxPw_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 192.968 | 0.697 | 58.025 | 19.147 | 27.254 | 26.601 | 3.848 |

## 2026-06-25 12:33:41 UTC | OkKEFnPQxPw_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/OkKEFnPQxPw_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `192.968` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.697 |
| save_clips | - |
| sample_frames | 1.500 |
| caption_frames | 42.737 |
| sample_fps | 2.316 |
| detect_object_yolo | 9.439 |
| audio_scan | 15.514 |
| asr_timings | 10.151 |
| ast_timings | 32.352 |
| describe_scenes | 19.147 |
| summarize_scenes | 27.254 |
| synthesize_synopsis | 26.601 |
| make_embedding | 3.848 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.243 |
| branch_yolo_total | 11.760 |
| branch_audio_total | 58.025 |
