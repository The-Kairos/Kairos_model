# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 11:53:55 UTC | jogh5IF2rEY_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 220.405 | 0.813 | 64.460 | 28.591 | 32.868 | 20.583 | 5.102 |

## 2026-06-26 11:53:55 UTC | jogh5IF2rEY_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jogh5IF2rEY_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `220.405` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.813 |
| save_clips | - |
| sample_frames | 1.324 |
| caption_frames | 52.145 |
| sample_fps | 2.430 |
| detect_object_yolo | 10.671 |
| audio_scan | 14.867 |
| asr_timings | 8.215 |
| ast_timings | 41.369 |
| describe_scenes | 28.591 |
| summarize_scenes | 32.868 |
| synthesize_synopsis | 20.583 |
| make_embedding | 5.102 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 53.475 |
| branch_yolo_total | 13.107 |
| branch_audio_total | 64.460 |
