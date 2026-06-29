# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 18:38:58 UTC | rUycc1YD41Q_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 155.976 | 0.809 | 51.843 | 16.776 | 13.818 | 14.496 | 3.574 |

## 2026-06-26 18:38:58 UTC | rUycc1YD41Q_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/rUycc1YD41Q_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `155.976` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.809 |
| save_clips | - |
| sample_frames | 1.219 |
| caption_frames | 40.253 |
| sample_fps | 2.306 |
| detect_object_yolo | 9.328 |
| audio_scan | 12.907 |
| asr_timings | 9.063 |
| ast_timings | 29.864 |
| describe_scenes | 16.776 |
| summarize_scenes | 13.818 |
| synthesize_synopsis | 14.496 |
| make_embedding | 3.574 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.478 |
| branch_yolo_total | 11.640 |
| branch_audio_total | 51.843 |
