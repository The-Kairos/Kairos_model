# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 11:15:39 UTC | NyOx_Dussuo_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 132.330 | 0.728 | 38.105 | 12.926 | 14.596 | 27.747 | 2.548 |

## 2026-06-25 11:15:39 UTC | NyOx_Dussuo_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/NyOx_Dussuo_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `132.330` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.728 |
| save_clips | - |
| sample_frames | 0.607 |
| caption_frames | 24.997 |
| sample_fps | 1.885 |
| detect_object_yolo | 6.791 |
| audio_scan | 10.512 |
| asr_timings | 8.614 |
| ast_timings | 18.971 |
| describe_scenes | 12.926 |
| summarize_scenes | 14.596 |
| synthesize_synopsis | 27.747 |
| make_embedding | 2.548 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 25.609 |
| branch_yolo_total | 8.682 |
| branch_audio_total | 38.105 |
