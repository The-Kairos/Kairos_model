# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 13:13:18 UTC | -ruy-w0bxvA_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 128.641 | 0.765 | 50.735 | 9.655 | 7.627 | 8.873 | 3.379 |

## 2026-06-27 13:13:18 UTC | -ruy-w0bxvA_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-ruy-w0bxvA_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `128.641` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.765 |
| save_clips | - |
| sample_frames | 0.791 |
| caption_frames | 35.085 |
| sample_fps | 2.133 |
| detect_object_yolo | 8.205 |
| audio_scan | 14.861 |
| asr_timings | 9.192 |
| ast_timings | 26.674 |
| describe_scenes | 9.655 |
| summarize_scenes | 7.627 |
| synthesize_synopsis | 8.873 |
| make_embedding | 3.379 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 35.882 |
| branch_yolo_total | 10.344 |
| branch_audio_total | 50.735 |
