# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 21:07:07 UTC | WlJGA2-wZQ4_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 173.511 | 0.778 | 104.027 | 9.701 | 6.447 | 8.536 | 2.755 |

## 2026-06-25 21:07:07 UTC | WlJGA2-wZQ4_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/WlJGA2-wZQ4_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `173.511` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.778 |
| save_clips | - |
| sample_frames | 0.881 |
| caption_frames | 29.230 |
| sample_fps | 2.054 |
| detect_object_yolo | 7.686 |
| audio_scan | 11.761 |
| asr_timings | 70.747 |
| ast_timings | 21.510 |
| describe_scenes | 9.701 |
| summarize_scenes | 6.447 |
| synthesize_synopsis | 8.536 |
| make_embedding | 2.755 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.116 |
| branch_yolo_total | 9.746 |
| branch_audio_total | 104.027 |
