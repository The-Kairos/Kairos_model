# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 19:39:35 UTC | Bjl5AomF9BY_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 140.022 | 0.825 | 45.273 | 14.282 | 10.697 | 13.060 | 3.719 |

## 2026-06-24 19:39:35 UTC | Bjl5AomF9BY_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Bjl5AomF9BY_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `140.022` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.825 |
| save_clips | - |
| sample_frames | 1.309 |
| caption_frames | 38.542 |
| sample_fps | 2.264 |
| detect_object_yolo | 8.650 |
| audio_scan | 8.624 |
| asr_timings | 6.686 |
| ast_timings | 29.954 |
| describe_scenes | 14.282 |
| summarize_scenes | 10.697 |
| synthesize_synopsis | 13.060 |
| make_embedding | 3.719 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.856 |
| branch_yolo_total | 10.920 |
| branch_audio_total | 45.273 |
