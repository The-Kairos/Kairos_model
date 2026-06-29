# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 01:04:07 UTC | bUa-0ptWL5M_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 118.291 | 0.784 | 44.994 | 10.146 | 14.191 | 8.456 | 2.582 |

## 2026-06-26 01:04:07 UTC | bUa-0ptWL5M_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/bUa-0ptWL5M_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `118.291` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.784 |
| save_clips | - |
| sample_frames | 0.562 |
| caption_frames | 26.045 |
| sample_fps | 2.002 |
| detect_object_yolo | 7.119 |
| audio_scan | 14.791 |
| asr_timings | 11.601 |
| ast_timings | 18.594 |
| describe_scenes | 10.146 |
| summarize_scenes | 14.191 |
| synthesize_synopsis | 8.456 |
| make_embedding | 2.582 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 26.613 |
| branch_yolo_total | 9.126 |
| branch_audio_total | 44.994 |
