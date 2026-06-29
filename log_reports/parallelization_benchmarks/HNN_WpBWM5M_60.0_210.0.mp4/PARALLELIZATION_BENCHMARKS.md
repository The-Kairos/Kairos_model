# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 03:26:19 UTC | HNN_WpBWM5M_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 139.876 | 0.771 | 52.746 | 10.029 | 8.900 | 8.642 | 3.609 |

## 2026-06-25 03:26:19 UTC | HNN_WpBWM5M_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/HNN_WpBWM5M_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `139.876` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.771 |
| save_clips | - |
| sample_frames | 1.065 |
| caption_frames | 41.556 |
| sample_fps | 2.251 |
| detect_object_yolo | 8.903 |
| audio_scan | 11.729 |
| asr_timings | 11.501 |
| ast_timings | 29.508 |
| describe_scenes | 10.029 |
| summarize_scenes | 8.900 |
| synthesize_synopsis | 8.642 |
| make_embedding | 3.609 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 42.627 |
| branch_yolo_total | 11.160 |
| branch_audio_total | 52.746 |
