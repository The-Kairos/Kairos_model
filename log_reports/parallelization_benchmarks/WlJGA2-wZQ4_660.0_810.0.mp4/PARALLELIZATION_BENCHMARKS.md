# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 21:16:24 UTC | WlJGA2-wZQ4_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 119.477 | 0.803 | 51.990 | 10.635 | 6.502 | 10.633 | 2.891 |

## 2026-06-25 21:16:24 UTC | WlJGA2-wZQ4_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/WlJGA2-wZQ4_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `119.477` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.803 |
| save_clips | - |
| sample_frames | 0.851 |
| caption_frames | 19.727 |
| sample_fps | 2.120 |
| detect_object_yolo | 7.097 |
| audio_scan | 13.637 |
| asr_timings | 16.395 |
| ast_timings | 21.950 |
| describe_scenes | 10.635 |
| summarize_scenes | 6.502 |
| synthesize_synopsis | 10.633 |
| make_embedding | 2.891 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 20.583 |
| branch_yolo_total | 9.223 |
| branch_audio_total | 51.990 |
