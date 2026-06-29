# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 02:16:13 UTC | vK4c5-_A9vs_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 199.228 | 0.797 | 70.026 | 17.119 | 11.881 | 12.643 | 5.645 |

## 2026-06-27 02:16:13 UTC | vK4c5-_A9vs_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/vK4c5-_A9vs_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `199.228` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.797 |
| save_clips | - |
| sample_frames | 1.528 |
| caption_frames | 64.133 |
| sample_fps | 2.618 |
| detect_object_yolo | 11.404 |
| audio_scan | 15.041 |
| asr_timings | 8.945 |
| ast_timings | 46.032 |
| describe_scenes | 17.119 |
| summarize_scenes | 11.881 |
| synthesize_synopsis | 12.643 |
| make_embedding | 5.645 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 65.667 |
| branch_yolo_total | 14.028 |
| branch_audio_total | 70.026 |
