# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 11:38:03 UTC | 5Ib6GnYyw-o_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 178.243 | 0.799 | 54.542 | 33.270 | 13.743 | 15.675 | 3.642 |

## 2026-06-24 11:38:03 UTC | 5Ib6GnYyw-o_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/5Ib6GnYyw-o_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `178.243` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.799 |
| save_clips | - |
| sample_frames | 1.325 |
| caption_frames | 41.891 |
| sample_fps | 2.298 |
| detect_object_yolo | 9.593 |
| audio_scan | 15.140 |
| asr_timings | 8.877 |
| ast_timings | 30.516 |
| describe_scenes | 33.270 |
| summarize_scenes | 13.743 |
| synthesize_synopsis | 15.675 |
| make_embedding | 3.642 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.223 |
| branch_yolo_total | 11.897 |
| branch_audio_total | 54.542 |
