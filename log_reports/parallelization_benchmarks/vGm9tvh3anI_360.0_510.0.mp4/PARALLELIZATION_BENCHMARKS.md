# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 02:07:17 UTC | vGm9tvh3anI_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 134.615 | 0.762 | 49.966 | 10.377 | 9.644 | 9.451 | 3.272 |

## 2026-06-27 02:07:17 UTC | vGm9tvh3anI_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/vGm9tvh3anI_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `134.615` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.762 |
| save_clips | - |
| sample_frames | 1.101 |
| caption_frames | 37.512 |
| sample_fps | 2.197 |
| detect_object_yolo | 8.902 |
| audio_scan | 12.948 |
| asr_timings | 10.046 |
| ast_timings | 26.964 |
| describe_scenes | 10.377 |
| summarize_scenes | 9.644 |
| synthesize_synopsis | 9.451 |
| make_embedding | 3.272 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.619 |
| branch_yolo_total | 11.105 |
| branch_audio_total | 49.966 |
