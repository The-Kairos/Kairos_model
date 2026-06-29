# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 02:50:39 UTC | wFbRG1IrNz0_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 146.701 | 0.760 | 57.464 | 11.337 | 7.378 | 7.884 | 4.108 |

## 2026-06-27 02:50:39 UTC | wFbRG1IrNz0_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/wFbRG1IrNz0_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `146.701` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.760 |
| save_clips | - |
| sample_frames | 1.178 |
| caption_frames | 43.700 |
| sample_fps | 2.304 |
| detect_object_yolo | 9.201 |
| audio_scan | 15.063 |
| asr_timings | 9.966 |
| ast_timings | 32.427 |
| describe_scenes | 11.337 |
| summarize_scenes | 7.378 |
| synthesize_synopsis | 7.884 |
| make_embedding | 4.108 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.883 |
| branch_yolo_total | 11.511 |
| branch_audio_total | 57.464 |
