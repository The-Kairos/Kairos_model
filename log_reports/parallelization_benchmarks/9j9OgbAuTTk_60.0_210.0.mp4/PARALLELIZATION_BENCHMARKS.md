# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 18:35:27 UTC | 9j9OgbAuTTk_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 221.267 | 0.848 | 66.066 | 28.627 | 28.742 | 20.603 | 5.036 |

## 2026-06-24 18:35:27 UTC | 9j9OgbAuTTk_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/9j9OgbAuTTk_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `221.267` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.848 |
| save_clips | - |
| sample_frames | 1.760 |
| caption_frames | 54.733 |
| sample_fps | 2.707 |
| detect_object_yolo | 10.728 |
| audio_scan | 8.613 |
| asr_timings | 15.946 |
| ast_timings | 41.499 |
| describe_scenes | 28.627 |
| summarize_scenes | 28.742 |
| synthesize_synopsis | 20.603 |
| make_embedding | 5.036 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 56.499 |
| branch_yolo_total | 13.440 |
| branch_audio_total | 66.066 |
