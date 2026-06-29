# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 12:43:10 UTC | 6ieHWdhGczs_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 230.300 | 0.712 | 84.520 | 19.129 | 37.690 | 26.481 | 3.977 |

## 2026-06-24 12:43:10 UTC | 6ieHWdhGczs_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/6ieHWdhGczs_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `230.300` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.712 |
| save_clips | - |
| sample_frames | 1.876 |
| caption_frames | 42.974 |
| sample_fps | 2.350 |
| detect_object_yolo | 9.202 |
| audio_scan | 13.802 |
| asr_timings | 39.166 |
| ast_timings | 31.544 |
| describe_scenes | 19.129 |
| summarize_scenes | 37.690 |
| synthesize_synopsis | 26.481 |
| make_embedding | 3.977 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.856 |
| branch_yolo_total | 11.558 |
| branch_audio_total | 84.520 |
