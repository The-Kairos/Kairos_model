# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 04:06:02 UTC | I96hjUvyk30_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 111.406 | 0.769 | 59.439 | 3.731 | 7.659 | 8.623 | 2.061 |

## 2026-06-25 04:06:02 UTC | I96hjUvyk30_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/I96hjUvyk30_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `111.406` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.769 |
| save_clips | - |
| sample_frames | 0.476 |
| caption_frames | 18.422 |
| sample_fps | 1.915 |
| detect_object_yolo | 6.913 |
| audio_scan | 13.809 |
| asr_timings | 32.570 |
| ast_timings | 13.051 |
| describe_scenes | 3.731 |
| summarize_scenes | 7.659 |
| synthesize_synopsis | 8.623 |
| make_embedding | 2.061 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 18.904 |
| branch_yolo_total | 8.834 |
| branch_audio_total | 59.439 |
