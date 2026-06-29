# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 20:02:14 UTC | VgmqAM03nQY_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 177.497 | 0.649 | 59.610 | 17.390 | 18.967 | 11.276 | 4.201 |

## 2026-06-25 20:02:14 UTC | VgmqAM03nQY_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/VgmqAM03nQY_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `177.497` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.649 |
| save_clips | - |
| sample_frames | 1.258 |
| caption_frames | 50.208 |
| sample_fps | 2.191 |
| detect_object_yolo | 10.295 |
| audio_scan | 13.976 |
| asr_timings | 9.922 |
| ast_timings | 35.704 |
| describe_scenes | 17.390 |
| summarize_scenes | 18.967 |
| synthesize_synopsis | 11.276 |
| make_embedding | 4.201 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.472 |
| branch_yolo_total | 12.492 |
| branch_audio_total | 59.610 |
