# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 04:51:19 UTC | ISVl0Xz2iuQ_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 171.401 | 0.687 | 58.711 | 18.275 | 8.410 | 10.537 | 5.073 |

## 2026-06-25 04:51:19 UTC | ISVl0Xz2iuQ_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ISVl0Xz2iuQ_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `171.401` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.687 |
| save_clips | - |
| sample_frames | 1.581 |
| caption_frames | 53.522 |
| sample_fps | 2.412 |
| detect_object_yolo | 10.803 |
| audio_scan | 8.570 |
| asr_timings | 8.826 |
| ast_timings | 41.307 |
| describe_scenes | 18.275 |
| summarize_scenes | 8.410 |
| synthesize_synopsis | 10.537 |
| make_embedding | 5.073 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 55.109 |
| branch_yolo_total | 13.220 |
| branch_audio_total | 58.711 |
