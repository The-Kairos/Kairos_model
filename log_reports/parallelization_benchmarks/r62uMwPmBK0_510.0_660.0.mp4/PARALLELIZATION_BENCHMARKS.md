# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 18:23:22 UTC | r62uMwPmBK0_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 140.235 | 0.733 | 58.045 | 20.518 | 24.413 | 15.039 | 5.027 |

## 2026-06-26 18:23:22 UTC | r62uMwPmBK0_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/r62uMwPmBK0_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `140.235` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.733 |
| save_clips | - |
| sample_frames | 1.779 |
| caption_frames | 56.260 |
| sample_fps | 2.527 |
| detect_object_yolo | 11.420 |
| audio_scan | 1.079 |
| asr_timings | 0.000 |
| ast_timings | 0.000 |
| describe_scenes | 20.518 |
| summarize_scenes | 24.413 |
| synthesize_synopsis | 15.039 |
| make_embedding | 5.027 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 58.045 |
| branch_yolo_total | 13.953 |
| branch_audio_total | 1.087 |
