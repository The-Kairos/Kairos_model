# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 19:27:04 UTC | s7MesIm6VLw_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 204.506 | 0.638 | 97.771 | 22.222 | 13.169 | 11.745 | 3.585 |

## 2026-06-26 19:27:04 UTC | s7MesIm6VLw_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/s7MesIm6VLw_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `204.506` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.638 |
| save_clips | - |
| sample_frames | 1.171 |
| caption_frames | 41.617 |
| sample_fps | 2.118 |
| detect_object_yolo | 9.037 |
| audio_scan | 12.643 |
| asr_timings | 55.563 |
| ast_timings | 29.556 |
| describe_scenes | 22.222 |
| summarize_scenes | 13.169 |
| synthesize_synopsis | 11.745 |
| make_embedding | 3.585 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 42.794 |
| branch_yolo_total | 11.161 |
| branch_audio_total | 97.771 |
