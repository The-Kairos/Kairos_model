# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 14:52:53 UTC | Pt5YUleXhTg_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 204.877 | 0.783 | 65.598 | 24.597 | 17.420 | 17.285 | 5.449 |

## 2026-06-25 14:52:53 UTC | Pt5YUleXhTg_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Pt5YUleXhTg_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `204.877` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.783 |
| save_clips | - |
| sample_frames | 1.413 |
| caption_frames | 57.297 |
| sample_fps | 2.524 |
| detect_object_yolo | 11.065 |
| audio_scan | 8.862 |
| asr_timings | 13.092 |
| ast_timings | 43.636 |
| describe_scenes | 24.597 |
| summarize_scenes | 17.420 |
| synthesize_synopsis | 17.285 |
| make_embedding | 5.449 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 58.716 |
| branch_yolo_total | 13.595 |
| branch_audio_total | 65.598 |
