# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 01:48:05 UTC | c1lX1gd5I78_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 191.500 | 0.848 | 68.067 | 16.035 | 20.807 | 7.171 | 5.544 |

## 2026-06-26 01:48:05 UTC | c1lX1gd5I78_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/c1lX1gd5I78_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `191.500` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.848 |
| save_clips | - |
| sample_frames | 1.496 |
| caption_frames | 56.618 |
| sample_fps | 2.493 |
| detect_object_yolo | 11.016 |
| audio_scan | 14.007 |
| asr_timings | 10.766 |
| ast_timings | 43.285 |
| describe_scenes | 16.035 |
| summarize_scenes | 20.807 |
| synthesize_synopsis | 7.171 |
| make_embedding | 5.544 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 58.120 |
| branch_yolo_total | 13.515 |
| branch_audio_total | 68.067 |
