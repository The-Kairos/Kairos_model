# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 05:14:15 UTC | gS7RgLgSw-o_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 230.507 | 0.840 | 114.189 | 15.958 | 12.349 | 15.421 | 4.484 |

## 2026-06-26 05:14:15 UTC | gS7RgLgSw-o_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/gS7RgLgSw-o_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `230.507` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.840 |
| save_clips | - |
| sample_frames | 1.336 |
| caption_frames | 51.318 |
| sample_fps | 2.450 |
| detect_object_yolo | 10.656 |
| audio_scan | 11.001 |
| asr_timings | 64.193 |
| ast_timings | 38.986 |
| describe_scenes | 15.958 |
| summarize_scenes | 12.349 |
| synthesize_synopsis | 15.421 |
| make_embedding | 4.484 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 52.660 |
| branch_yolo_total | 13.112 |
| branch_audio_total | 114.189 |
