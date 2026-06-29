# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 00:36:11 UTC | Fptgkh2-2DQ_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 152.529 | 0.799 | 54.803 | 12.785 | 13.757 | 9.760 | 3.786 |

## 2026-06-25 00:36:11 UTC | Fptgkh2-2DQ_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Fptgkh2-2DQ_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `152.529` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.799 |
| save_clips | - |
| sample_frames | 1.131 |
| caption_frames | 42.692 |
| sample_fps | 2.267 |
| detect_object_yolo | 9.319 |
| audio_scan | 13.903 |
| asr_timings | 8.192 |
| ast_timings | 32.700 |
| describe_scenes | 12.785 |
| summarize_scenes | 13.757 |
| synthesize_synopsis | 9.760 |
| make_embedding | 3.786 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.828 |
| branch_yolo_total | 11.592 |
| branch_audio_total | 54.803 |
