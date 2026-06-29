# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 19:48:42 UTC | BlZaRZqMZ84_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 132.796 | 0.638 | 46.352 | 8.963 | 21.081 | 10.554 | 3.233 |

## 2026-06-24 19:48:42 UTC | BlZaRZqMZ84_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/BlZaRZqMZ84_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `132.796` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.638 |
| save_clips | - |
| sample_frames | 0.746 |
| caption_frames | 29.124 |
| sample_fps | 1.955 |
| detect_object_yolo | 8.754 |
| audio_scan | 14.904 |
| asr_timings | 7.466 |
| ast_timings | 23.974 |
| describe_scenes | 8.963 |
| summarize_scenes | 21.081 |
| synthesize_synopsis | 10.554 |
| make_embedding | 3.233 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 29.875 |
| branch_yolo_total | 10.715 |
| branch_audio_total | 46.352 |
