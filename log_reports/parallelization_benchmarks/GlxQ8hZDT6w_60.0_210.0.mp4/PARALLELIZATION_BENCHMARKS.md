# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 02:13:40 UTC | GlxQ8hZDT6w_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 147.872 | 0.802 | 63.601 | 11.303 | 7.044 | 9.469 | 3.616 |

## 2026-06-25 02:13:40 UTC | GlxQ8hZDT6w_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/GlxQ8hZDT6w_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `147.872` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.802 |
| save_clips | - |
| sample_frames | 1.238 |
| caption_frames | 38.046 |
| sample_fps | 2.316 |
| detect_object_yolo | 9.021 |
| audio_scan | 14.906 |
| asr_timings | 19.408 |
| ast_timings | 29.278 |
| describe_scenes | 11.303 |
| summarize_scenes | 7.044 |
| synthesize_synopsis | 9.469 |
| make_embedding | 3.616 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.290 |
| branch_yolo_total | 11.342 |
| branch_audio_total | 63.601 |
