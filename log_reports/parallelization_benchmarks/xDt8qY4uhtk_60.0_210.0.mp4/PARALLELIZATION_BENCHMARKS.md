# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 03:41:47 UTC | xDt8qY4uhtk_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 137.140 | 0.815 | 54.588 | 9.405 | 7.432 | 6.329 | 3.569 |

## 2026-06-27 03:41:47 UTC | xDt8qY4uhtk_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/xDt8qY4uhtk_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `137.140` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.815 |
| save_clips | - |
| sample_frames | 1.024 |
| caption_frames | 40.752 |
| sample_fps | 2.300 |
| detect_object_yolo | 9.444 |
| audio_scan | 14.245 |
| asr_timings | 9.683 |
| ast_timings | 30.651 |
| describe_scenes | 9.405 |
| summarize_scenes | 7.432 |
| synthesize_synopsis | 6.329 |
| make_embedding | 3.569 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.782 |
| branch_yolo_total | 11.750 |
| branch_audio_total | 54.588 |
