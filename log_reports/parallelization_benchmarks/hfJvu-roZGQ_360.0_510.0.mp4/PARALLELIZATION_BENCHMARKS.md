# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 06:31:40 UTC | hfJvu-roZGQ_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 182.748 | 0.628 | 79.633 | 13.738 | 15.050 | 19.705 | 3.319 |

## 2026-06-26 06:31:40 UTC | hfJvu-roZGQ_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/hfJvu-roZGQ_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `182.748` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.628 |
| save_clips | - |
| sample_frames | 0.759 |
| caption_frames | 38.825 |
| sample_fps | 0.917 |
| detect_object_yolo | 8.761 |
| audio_scan | 14.952 |
| asr_timings | 38.191 |
| ast_timings | 26.472 |
| describe_scenes | 13.738 |
| summarize_scenes | 15.050 |
| synthesize_synopsis | 19.705 |
| make_embedding | 3.319 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.590 |
| branch_yolo_total | 9.684 |
| branch_audio_total | 79.633 |
