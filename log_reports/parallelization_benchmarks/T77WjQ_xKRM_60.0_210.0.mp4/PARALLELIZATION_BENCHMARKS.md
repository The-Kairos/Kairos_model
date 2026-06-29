# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 17:44:24 UTC | T77WjQ_xKRM_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 89.979 | 0.677 | 34.849 | 4.686 | 7.598 | 14.620 | 1.801 |

## 2026-06-25 17:44:24 UTC | T77WjQ_xKRM_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/T77WjQ_xKRM_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `89.979` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.677 |
| save_clips | - |
| sample_frames | 0.338 |
| caption_frames | 15.597 |
| sample_fps | 1.719 |
| detect_object_yolo | 6.702 |
| audio_scan | 14.928 |
| asr_timings | 9.687 |
| ast_timings | 10.226 |
| describe_scenes | 4.686 |
| summarize_scenes | 7.598 |
| synthesize_synopsis | 14.620 |
| make_embedding | 1.801 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 15.941 |
| branch_yolo_total | 8.427 |
| branch_audio_total | 34.849 |
