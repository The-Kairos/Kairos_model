# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 20:06:25 UTC | VgmqAM03nQY_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 250.169 | 0.668 | 111.696 | 21.280 | 18.507 | 9.403 | 6.077 |

## 2026-06-25 20:06:25 UTC | VgmqAM03nQY_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/VgmqAM03nQY_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `250.169` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.668 |
| save_clips | - |
| sample_frames | 1.790 |
| caption_frames | 64.761 |
| sample_fps | 2.543 |
| detect_object_yolo | 12.025 |
| audio_scan | 16.073 |
| asr_timings | 46.301 |
| ast_timings | 49.314 |
| describe_scenes | 21.280 |
| summarize_scenes | 18.507 |
| synthesize_synopsis | 9.403 |
| make_embedding | 6.077 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 66.557 |
| branch_yolo_total | 14.573 |
| branch_audio_total | 111.696 |
