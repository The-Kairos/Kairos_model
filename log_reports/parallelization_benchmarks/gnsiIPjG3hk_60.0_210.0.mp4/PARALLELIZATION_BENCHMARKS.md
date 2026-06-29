# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 05:43:59 UTC | gnsiIPjG3hk_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 187.567 | 0.785 | 55.244 | 16.156 | 23.054 | 19.918 | 5.082 |

## 2026-06-26 05:43:59 UTC | gnsiIPjG3hk_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/gnsiIPjG3hk_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `187.567` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.785 |
| save_clips | - |
| sample_frames | 1.423 |
| caption_frames | 51.143 |
| sample_fps | 2.536 |
| detect_object_yolo | 10.801 |
| audio_scan | 6.566 |
| asr_timings | 6.771 |
| ast_timings | 41.899 |
| describe_scenes | 16.156 |
| summarize_scenes | 23.054 |
| synthesize_synopsis | 19.918 |
| make_embedding | 5.082 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 52.572 |
| branch_yolo_total | 13.343 |
| branch_audio_total | 55.244 |
