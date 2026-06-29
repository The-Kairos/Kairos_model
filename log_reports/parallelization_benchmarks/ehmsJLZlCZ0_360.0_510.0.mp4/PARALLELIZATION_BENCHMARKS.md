# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 03:54:33 UTC | ehmsJLZlCZ0_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 212.450 | 0.657 | 106.727 | 10.967 | 10.272 | 15.830 | 4.147 |

## 2026-06-26 03:54:33 UTC | ehmsJLZlCZ0_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ehmsJLZlCZ0_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `212.450` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.657 |
| save_clips | - |
| sample_frames | 1.436 |
| caption_frames | 48.720 |
| sample_fps | 2.295 |
| detect_object_yolo | 9.958 |
| audio_scan | 13.054 |
| asr_timings | 57.268 |
| ast_timings | 36.396 |
| describe_scenes | 10.967 |
| summarize_scenes | 10.272 |
| synthesize_synopsis | 15.830 |
| make_embedding | 4.147 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 50.162 |
| branch_yolo_total | 12.258 |
| branch_audio_total | 106.727 |
