# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 00:23:35 UTC | u5zzeSEJ0Ak_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 182.445 | 0.677 | 64.782 | 14.306 | 8.524 | 7.519 | 6.227 |

## 2026-06-27 00:23:35 UTC | u5zzeSEJ0Ak_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/u5zzeSEJ0Ak_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `182.445` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.677 |
| save_clips | - |
| sample_frames | 1.688 |
| caption_frames | 63.087 |
| sample_fps | 2.491 |
| detect_object_yolo | 12.295 |
| audio_scan | 6.517 |
| asr_timings | 10.235 |
| ast_timings | 47.391 |
| describe_scenes | 14.306 |
| summarize_scenes | 8.524 |
| synthesize_synopsis | 7.519 |
| make_embedding | 6.227 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 64.782 |
| branch_yolo_total | 14.792 |
| branch_audio_total | 64.151 |
