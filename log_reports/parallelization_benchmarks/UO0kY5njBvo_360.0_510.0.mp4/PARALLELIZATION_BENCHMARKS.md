# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 18:55:55 UTC | UO0kY5njBvo_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 135.531 | 0.780 | 50.143 | 12.470 | 6.847 | 16.801 | 3.066 |

## 2026-06-25 18:55:55 UTC | UO0kY5njBvo_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/UO0kY5njBvo_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `135.531` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.780 |
| save_clips | - |
| sample_frames | 1.062 |
| caption_frames | 32.631 |
| sample_fps | 2.152 |
| detect_object_yolo | 8.185 |
| audio_scan | 13.903 |
| asr_timings | 11.472 |
| ast_timings | 24.760 |
| describe_scenes | 12.470 |
| summarize_scenes | 6.847 |
| synthesize_synopsis | 16.801 |
| make_embedding | 3.066 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 33.699 |
| branch_yolo_total | 10.343 |
| branch_audio_total | 50.143 |
