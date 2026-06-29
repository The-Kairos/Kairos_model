# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 23:33:57 UTC | tLy1eqDOrMU_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 176.269 | 0.781 | 98.486 | 9.678 | 9.657 | 11.508 | 3.016 |

## 2026-06-26 23:33:57 UTC | tLy1eqDOrMU_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/tLy1eqDOrMU_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `176.269` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.781 |
| save_clips | - |
| sample_frames | 0.900 |
| caption_frames | 31.284 |
| sample_fps | 2.137 |
| detect_object_yolo | 7.413 |
| audio_scan | 12.829 |
| asr_timings | 60.930 |
| ast_timings | 24.719 |
| describe_scenes | 9.678 |
| summarize_scenes | 9.657 |
| synthesize_synopsis | 11.508 |
| make_embedding | 3.016 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.190 |
| branch_yolo_total | 9.555 |
| branch_audio_total | 98.486 |
