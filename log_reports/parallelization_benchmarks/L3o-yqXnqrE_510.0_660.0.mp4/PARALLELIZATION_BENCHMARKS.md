# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 07:09:09 UTC | L3o-yqXnqrE_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 185.717 | 0.647 | 54.734 | 18.627 | 21.844 | 28.705 | 3.619 |

## 2026-06-25 07:09:09 UTC | L3o-yqXnqrE_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/L3o-yqXnqrE_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `185.717` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.647 |
| save_clips | - |
| sample_frames | 1.177 |
| caption_frames | 43.550 |
| sample_fps | 2.138 |
| detect_object_yolo | 9.216 |
| audio_scan | 13.839 |
| asr_timings | 11.118 |
| ast_timings | 29.768 |
| describe_scenes | 18.627 |
| summarize_scenes | 21.844 |
| synthesize_synopsis | 28.705 |
| make_embedding | 3.619 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.733 |
| branch_yolo_total | 11.360 |
| branch_audio_total | 54.734 |
