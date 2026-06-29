# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 23:48:33 UTC | _UxECICm2eY_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 107.042 | 0.655 | 42.145 | 7.230 | 6.590 | 6.997 | 2.742 |

## 2026-06-25 23:48:33 UTC | _UxECICm2eY_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/_UxECICm2eY_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `107.042` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.655 |
| save_clips | - |
| sample_frames | 0.784 |
| caption_frames | 28.943 |
| sample_fps | 1.943 |
| detect_object_yolo | 7.610 |
| audio_scan | 9.563 |
| asr_timings | 10.634 |
| ast_timings | 21.939 |
| describe_scenes | 7.230 |
| summarize_scenes | 6.590 |
| synthesize_synopsis | 6.997 |
| make_embedding | 2.742 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 29.733 |
| branch_yolo_total | 9.558 |
| branch_audio_total | 42.145 |
