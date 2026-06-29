# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 19:56:07 UTC | VayyLoioSAk_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 185.879 | 0.640 | 69.816 | 17.445 | 9.431 | 9.413 | 5.411 |

## 2026-06-25 19:56:07 UTC | VayyLoioSAk_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/VayyLoioSAk_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `185.879` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.640 |
| save_clips | - |
| sample_frames | 1.316 |
| caption_frames | 57.305 |
| sample_fps | 2.283 |
| detect_object_yolo | 11.408 |
| audio_scan | 16.091 |
| asr_timings | 10.720 |
| ast_timings | 42.996 |
| describe_scenes | 17.445 |
| summarize_scenes | 9.431 |
| synthesize_synopsis | 9.413 |
| make_embedding | 5.411 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 58.626 |
| branch_yolo_total | 13.696 |
| branch_audio_total | 69.816 |
