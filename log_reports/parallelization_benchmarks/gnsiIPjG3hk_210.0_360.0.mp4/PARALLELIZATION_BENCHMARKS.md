# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 05:32:33 UTC | gnsiIPjG3hk_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 183.234 | 0.782 | 59.140 | 14.221 | 21.288 | 17.167 | 4.615 |

## 2026-06-26 05:32:33 UTC | gnsiIPjG3hk_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/gnsiIPjG3hk_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `183.234` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.782 |
| save_clips | - |
| sample_frames | 1.286 |
| caption_frames | 50.685 |
| sample_fps | 2.456 |
| detect_object_yolo | 10.188 |
| audio_scan | 9.805 |
| asr_timings | 10.458 |
| ast_timings | 38.869 |
| describe_scenes | 14.221 |
| summarize_scenes | 21.288 |
| synthesize_synopsis | 17.167 |
| make_embedding | 4.615 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.976 |
| branch_yolo_total | 12.650 |
| branch_audio_total | 59.140 |
