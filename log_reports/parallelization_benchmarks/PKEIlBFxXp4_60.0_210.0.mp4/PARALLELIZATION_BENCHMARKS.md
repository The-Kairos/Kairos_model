# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 13:25:42 UTC | PKEIlBFxXp4_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 211.391 | 0.657 | 63.622 | 32.942 | 17.439 | 29.264 | 4.592 |

## 2026-06-25 13:25:42 UTC | PKEIlBFxXp4_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/PKEIlBFxXp4_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `211.391` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.657 |
| save_clips | - |
| sample_frames | 1.339 |
| caption_frames | 48.030 |
| sample_fps | 2.277 |
| detect_object_yolo | 9.805 |
| audio_scan | 15.517 |
| asr_timings | 11.820 |
| ast_timings | 36.277 |
| describe_scenes | 32.942 |
| summarize_scenes | 17.439 |
| synthesize_synopsis | 29.264 |
| make_embedding | 4.592 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.375 |
| branch_yolo_total | 12.088 |
| branch_audio_total | 63.622 |
