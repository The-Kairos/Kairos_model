# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 08:03:58 UTC | pMWJa4dYbkg_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 166.779 | 0.776 | 61.529 | 11.340 | 10.280 | 13.299 | 4.469 |

## 2026-06-28 08:03:58 UTC | pMWJa4dYbkg_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/pMWJa4dYbkg_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `166.779` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.776 |
| save_clips | - |
| sample_frames | 1.459 |
| caption_frames | 49.550 |
| sample_fps | 2.416 |
| detect_object_yolo | 10.261 |
| audio_scan | 14.879 |
| asr_timings | 9.213 |
| ast_timings | 37.428 |
| describe_scenes | 11.340 |
| summarize_scenes | 10.280 |
| synthesize_synopsis | 13.299 |
| make_embedding | 4.469 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.015 |
| branch_yolo_total | 12.683 |
| branch_audio_total | 61.529 |
