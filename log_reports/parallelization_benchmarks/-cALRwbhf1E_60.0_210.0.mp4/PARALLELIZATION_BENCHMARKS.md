# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 08:41:30 UTC | -cALRwbhf1E_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 195.071 | 0.820 | 58.377 | 24.018 | 19.967 | 23.861 | 4.287 |

## 2026-06-24 08:41:30 UTC | -cALRwbhf1E_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-cALRwbhf1E_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `195.071` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.820 |
| save_clips | - |
| sample_frames | 1.518 |
| caption_frames | 48.599 |
| sample_fps | 2.435 |
| detect_object_yolo | 9.836 |
| audio_scan | 13.880 |
| asr_timings | 9.061 |
| ast_timings | 35.427 |
| describe_scenes | 24.018 |
| summarize_scenes | 19.967 |
| synthesize_synopsis | 23.861 |
| make_embedding | 4.287 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 50.123 |
| branch_yolo_total | 12.278 |
| branch_audio_total | 58.377 |
