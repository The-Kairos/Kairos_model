# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 01:52:26 UTC | GZi8Gl6teBk_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 117.437 | 0.650 | 44.865 | 7.274 | 9.291 | 10.473 | 2.813 |

## 2026-06-25 01:52:26 UTC | GZi8Gl6teBk_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/GZi8Gl6teBk_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `117.437` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.650 |
| save_clips | - |
| sample_frames | 0.985 |
| caption_frames | 29.729 |
| sample_fps | 2.022 |
| detect_object_yolo | 7.948 |
| audio_scan | 13.789 |
| asr_timings | 9.743 |
| ast_timings | 21.325 |
| describe_scenes | 7.274 |
| summarize_scenes | 9.291 |
| synthesize_synopsis | 10.473 |
| make_embedding | 2.813 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.719 |
| branch_yolo_total | 9.975 |
| branch_audio_total | 44.865 |
