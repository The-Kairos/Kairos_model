# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 16:41:01 UTC | n_62dLVgd3I_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 74.126 | 0.637 | 41.116 | 4.456 | 2.309 | 7.051 | 1.351 |

## 2026-06-27 16:41:01 UTC | n_62dLVgd3I_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/n_62dLVgd3I_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `74.126` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.637 |
| save_clips | - |
| sample_frames | 0.103 |
| caption_frames | 8.299 |
| sample_fps | 1.647 |
| detect_object_yolo | 5.778 |
| audio_scan | 10.649 |
| asr_timings | 25.677 |
| ast_timings | 4.781 |
| describe_scenes | 4.456 |
| summarize_scenes | 2.309 |
| synthesize_synopsis | 7.051 |
| make_embedding | 1.351 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 8.408 |
| branch_yolo_total | 7.431 |
| branch_audio_total | 41.116 |
