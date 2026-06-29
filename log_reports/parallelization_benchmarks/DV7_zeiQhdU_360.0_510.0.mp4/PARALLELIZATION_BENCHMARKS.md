# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 22:56:18 UTC | DV7_zeiQhdU_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 136.872 | 0.782 | 54.700 | 13.037 | 6.948 | 6.205 | 3.555 |

## 2026-06-24 22:56:18 UTC | DV7_zeiQhdU_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/DV7_zeiQhdU_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `136.872` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.782 |
| save_clips | - |
| sample_frames | 0.912 |
| caption_frames | 38.190 |
| sample_fps | 2.216 |
| detect_object_yolo | 8.915 |
| audio_scan | 14.972 |
| asr_timings | 10.171 |
| ast_timings | 29.547 |
| describe_scenes | 13.037 |
| summarize_scenes | 6.948 |
| synthesize_synopsis | 6.205 |
| make_embedding | 3.555 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.108 |
| branch_yolo_total | 11.136 |
| branch_audio_total | 54.700 |
