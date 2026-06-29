# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 23:46:45 UTC | _UxECICm2eY_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 103.571 | 0.677 | 39.451 | 6.245 | 8.135 | 11.867 | 2.517 |

## 2026-06-25 23:46:45 UTC | _UxECICm2eY_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/_UxECICm2eY_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `103.571` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.677 |
| save_clips | - |
| sample_frames | 0.586 |
| caption_frames | 24.099 |
| sample_fps | 1.872 |
| detect_object_yolo | 6.719 |
| audio_scan | 11.861 |
| asr_timings | 11.460 |
| ast_timings | 16.120 |
| describe_scenes | 6.245 |
| summarize_scenes | 8.135 |
| synthesize_synopsis | 11.867 |
| make_embedding | 2.517 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 24.690 |
| branch_yolo_total | 8.597 |
| branch_audio_total | 39.451 |
