# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 19:27:08 UTC | B0GEURtHWyI_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 156.760 | 0.798 | 57.197 | 13.926 | 10.405 | 11.273 | 4.062 |

## 2026-06-24 19:27:08 UTC | B0GEURtHWyI_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/B0GEURtHWyI_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `156.760` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.798 |
| save_clips | - |
| sample_frames | 1.274 |
| caption_frames | 44.468 |
| sample_fps | 2.355 |
| detect_object_yolo | 9.576 |
| audio_scan | 14.008 |
| asr_timings | 10.159 |
| ast_timings | 33.021 |
| describe_scenes | 13.926 |
| summarize_scenes | 10.405 |
| synthesize_synopsis | 11.273 |
| make_embedding | 4.062 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.748 |
| branch_yolo_total | 11.937 |
| branch_audio_total | 57.197 |
