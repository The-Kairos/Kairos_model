# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 03:50:23 UTC | Hdfmhdj3Efg_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 154.504 | 0.850 | 64.842 | 12.215 | 8.917 | 13.840 | 3.314 |

## 2026-06-25 03:50:23 UTC | Hdfmhdj3Efg_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Hdfmhdj3Efg_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `154.504` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.850 |
| save_clips | - |
| sample_frames | 1.437 |
| caption_frames | 37.086 |
| sample_fps | 2.355 |
| detect_object_yolo | 8.260 |
| audio_scan | 13.818 |
| asr_timings | 24.104 |
| ast_timings | 26.913 |
| describe_scenes | 12.215 |
| summarize_scenes | 8.917 |
| synthesize_synopsis | 13.840 |
| make_embedding | 3.314 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.530 |
| branch_yolo_total | 10.622 |
| branch_audio_total | 64.842 |
