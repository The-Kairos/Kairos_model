# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 01:12:17 UTC | uWgykeg3gug_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 162.649 | 0.778 | 58.230 | 16.498 | 9.309 | 10.850 | 4.146 |

## 2026-06-27 01:12:17 UTC | uWgykeg3gug_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/uWgykeg3gug_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `162.649` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.778 |
| save_clips | - |
| sample_frames | 1.472 |
| caption_frames | 47.825 |
| sample_fps | 2.372 |
| detect_object_yolo | 9.735 |
| audio_scan | 13.945 |
| asr_timings | 9.723 |
| ast_timings | 34.553 |
| describe_scenes | 16.498 |
| summarize_scenes | 9.309 |
| synthesize_synopsis | 10.850 |
| make_embedding | 4.146 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.303 |
| branch_yolo_total | 12.112 |
| branch_audio_total | 58.230 |
