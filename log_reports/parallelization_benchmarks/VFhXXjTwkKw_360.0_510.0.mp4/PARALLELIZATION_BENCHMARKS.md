# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 19:40:23 UTC | VFhXXjTwkKw_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 135.777 | 0.818 | 40.986 | 13.032 | 19.575 | 12.759 | 3.037 |

## 2026-06-25 19:40:23 UTC | VFhXXjTwkKw_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/VFhXXjTwkKw_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `135.777` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.818 |
| save_clips | - |
| sample_frames | 0.841 |
| caption_frames | 33.015 |
| sample_fps | 2.172 |
| detect_object_yolo | 8.146 |
| audio_scan | 8.608 |
| asr_timings | 7.245 |
| ast_timings | 25.123 |
| describe_scenes | 13.032 |
| summarize_scenes | 19.575 |
| synthesize_synopsis | 12.759 |
| make_embedding | 3.037 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 33.862 |
| branch_yolo_total | 10.324 |
| branch_audio_total | 40.986 |
