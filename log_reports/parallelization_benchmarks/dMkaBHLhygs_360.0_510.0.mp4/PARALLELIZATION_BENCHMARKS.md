# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 02:57:32 UTC | dMkaBHLhygs_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 189.386 | 0.699 | 70.474 | 13.298 | 14.568 | 11.692 | 5.396 |

## 2026-06-26 02:57:32 UTC | dMkaBHLhygs_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/dMkaBHLhygs_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `189.386` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.699 |
| save_clips | - |
| sample_frames | 1.754 |
| caption_frames | 55.942 |
| sample_fps | 2.534 |
| detect_object_yolo | 11.544 |
| audio_scan | 16.461 |
| asr_timings | 9.256 |
| ast_timings | 44.748 |
| describe_scenes | 13.298 |
| summarize_scenes | 14.568 |
| synthesize_synopsis | 11.692 |
| make_embedding | 5.396 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 57.702 |
| branch_yolo_total | 14.083 |
| branch_audio_total | 70.474 |
