# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 18:06:18 UTC | 9_UC5c7LFN4_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 109.442 | 0.781 | 38.902 | 10.285 | 7.002 | 15.128 | 2.592 |

## 2026-06-24 18:06:18 UTC | 9_UC5c7LFN4_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/9_UC5c7LFN4_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `109.442` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.781 |
| save_clips | - |
| sample_frames | 0.617 |
| caption_frames | 23.872 |
| sample_fps | 2.018 |
| detect_object_yolo | 6.806 |
| audio_scan | 6.536 |
| asr_timings | 14.442 |
| ast_timings | 17.915 |
| describe_scenes | 10.285 |
| summarize_scenes | 7.002 |
| synthesize_synopsis | 15.128 |
| make_embedding | 2.592 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 24.495 |
| branch_yolo_total | 8.829 |
| branch_audio_total | 38.902 |
