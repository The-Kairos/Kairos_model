# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 13:19:09 UTC | PKEIlBFxXp4_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 173.052 | 0.664 | 50.976 | 29.563 | 16.934 | 16.461 | 3.599 |

## 2026-06-25 13:19:09 UTC | PKEIlBFxXp4_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/PKEIlBFxXp4_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `173.052` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.664 |
| save_clips | - |
| sample_frames | 1.171 |
| caption_frames | 41.174 |
| sample_fps | 2.121 |
| detect_object_yolo | 8.969 |
| audio_scan | 11.027 |
| asr_timings | 9.610 |
| ast_timings | 30.330 |
| describe_scenes | 29.563 |
| summarize_scenes | 16.934 |
| synthesize_synopsis | 16.461 |
| make_embedding | 3.599 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 42.351 |
| branch_yolo_total | 11.095 |
| branch_audio_total | 50.976 |
