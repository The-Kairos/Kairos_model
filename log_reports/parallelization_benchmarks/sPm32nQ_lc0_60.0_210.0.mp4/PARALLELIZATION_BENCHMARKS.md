# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 20:01:14 UTC | sPm32nQ_lc0_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 153.212 | 0.810 | 54.132 | 15.584 | 9.496 | 14.967 | 3.559 |

## 2026-06-26 20:01:14 UTC | sPm32nQ_lc0_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/sPm32nQ_lc0_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `153.212` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.810 |
| save_clips | - |
| sample_frames | 1.308 |
| caption_frames | 40.563 |
| sample_fps | 2.368 |
| detect_object_yolo | 9.015 |
| audio_scan | 14.987 |
| asr_timings | 9.678 |
| ast_timings | 29.459 |
| describe_scenes | 15.584 |
| summarize_scenes | 9.496 |
| synthesize_synopsis | 14.967 |
| make_embedding | 3.559 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.878 |
| branch_yolo_total | 11.389 |
| branch_audio_total | 54.132 |
