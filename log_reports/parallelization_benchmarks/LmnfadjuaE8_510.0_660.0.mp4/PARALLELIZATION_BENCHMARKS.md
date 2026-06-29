# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 07:45:12 UTC | LmnfadjuaE8_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 108.385 | 0.563 | 40.676 | 5.041 | 18.474 | 23.383 | 1.563 |

## 2026-06-25 07:45:12 UTC | LmnfadjuaE8_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/LmnfadjuaE8_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `108.385` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.563 |
| save_clips | - |
| sample_frames | 0.186 |
| caption_frames | 10.472 |
| sample_fps | 1.458 |
| detect_object_yolo | 5.091 |
| audio_scan | 15.821 |
| asr_timings | 17.852 |
| ast_timings | 6.995 |
| describe_scenes | 5.041 |
| summarize_scenes | 18.474 |
| synthesize_synopsis | 23.383 |
| make_embedding | 1.563 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 10.664 |
| branch_yolo_total | 6.555 |
| branch_audio_total | 40.676 |
