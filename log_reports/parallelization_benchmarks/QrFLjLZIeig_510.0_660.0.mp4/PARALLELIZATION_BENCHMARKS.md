# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 15:36:49 UTC | QrFLjLZIeig_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 212.507 | 0.798 | 60.719 | 21.965 | 34.082 | 23.593 | 4.548 |

## 2026-06-25 15:36:49 UTC | QrFLjLZIeig_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/QrFLjLZIeig_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `212.507` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.798 |
| save_clips | - |
| sample_frames | 1.417 |
| caption_frames | 50.894 |
| sample_fps | 2.478 |
| detect_object_yolo | 10.533 |
| audio_scan | 12.460 |
| asr_timings | 9.149 |
| ast_timings | 39.103 |
| describe_scenes | 21.965 |
| summarize_scenes | 34.082 |
| synthesize_synopsis | 23.593 |
| make_embedding | 4.548 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 52.317 |
| branch_yolo_total | 13.017 |
| branch_audio_total | 60.719 |
