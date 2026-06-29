# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 03:00:24 UTC | dMkaBHLhygs_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 171.108 | 0.694 | 56.737 | 13.456 | 16.519 | 22.776 | 3.878 |

## 2026-06-26 03:00:24 UTC | dMkaBHLhygs_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/dMkaBHLhygs_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `171.108` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.694 |
| save_clips | - |
| sample_frames | 1.234 |
| caption_frames | 42.324 |
| sample_fps | 2.226 |
| detect_object_yolo | 9.784 |
| audio_scan | 13.201 |
| asr_timings | 10.347 |
| ast_timings | 33.182 |
| describe_scenes | 13.456 |
| summarize_scenes | 16.519 |
| synthesize_synopsis | 22.776 |
| make_embedding | 3.878 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.564 |
| branch_yolo_total | 12.017 |
| branch_audio_total | 56.737 |
