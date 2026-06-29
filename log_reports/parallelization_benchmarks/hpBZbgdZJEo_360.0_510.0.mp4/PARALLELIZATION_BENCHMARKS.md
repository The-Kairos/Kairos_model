# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 07:13:57 UTC | hpBZbgdZJEo_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 177.341 | 0.841 | 55.242 | 21.252 | 17.849 | 20.598 | 3.965 |

## 2026-06-26 07:13:57 UTC | hpBZbgdZJEo_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/hpBZbgdZJEo_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `177.341` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.841 |
| save_clips | - |
| sample_frames | 1.443 |
| caption_frames | 42.519 |
| sample_fps | 2.424 |
| detect_object_yolo | 9.745 |
| audio_scan | 14.138 |
| asr_timings | 8.022 |
| ast_timings | 33.074 |
| describe_scenes | 21.252 |
| summarize_scenes | 17.849 |
| synthesize_synopsis | 20.598 |
| make_embedding | 3.965 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.968 |
| branch_yolo_total | 12.175 |
| branch_audio_total | 55.242 |
