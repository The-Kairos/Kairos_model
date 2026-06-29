# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 14:34:31 UTC | l3wh1vfwUO0_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 249.055 | 0.801 | 76.708 | 30.915 | 45.758 | 17.226 | 5.087 |

## 2026-06-26 14:34:31 UTC | l3wh1vfwUO0_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/l3wh1vfwUO0_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `249.055` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.801 |
| save_clips | - |
| sample_frames | 1.849 |
| caption_frames | 56.336 |
| sample_fps | 2.571 |
| detect_object_yolo | 10.389 |
| audio_scan | 13.007 |
| asr_timings | 21.481 |
| ast_timings | 42.212 |
| describe_scenes | 30.915 |
| summarize_scenes | 45.758 |
| synthesize_synopsis | 17.226 |
| make_embedding | 5.087 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 58.191 |
| branch_yolo_total | 12.966 |
| branch_audio_total | 76.708 |
