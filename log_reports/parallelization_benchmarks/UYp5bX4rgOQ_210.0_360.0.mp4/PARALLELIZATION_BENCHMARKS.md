# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 19:11:27 UTC | UYp5bX4rgOQ_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 209.743 | 0.827 | 73.279 | 25.909 | 12.587 | 13.860 | 5.681 |

## 2026-06-25 19:11:27 UTC | UYp5bX4rgOQ_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/UYp5bX4rgOQ_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `209.743` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.827 |
| save_clips | - |
| sample_frames | 1.834 |
| caption_frames | 59.974 |
| sample_fps | 2.695 |
| detect_object_yolo | 11.694 |
| audio_scan | 16.066 |
| asr_timings | 10.263 |
| ast_timings | 46.941 |
| describe_scenes | 25.909 |
| summarize_scenes | 12.587 |
| synthesize_synopsis | 13.860 |
| make_embedding | 5.681 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 61.814 |
| branch_yolo_total | 14.394 |
| branch_audio_total | 73.279 |
