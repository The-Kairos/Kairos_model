# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 20:21:59 UTC | COXt_GfXa2M_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 116.334 | 0.660 | 38.672 | 11.376 | 14.072 | 12.357 | 2.503 |

## 2026-06-24 20:21:59 UTC | COXt_GfXa2M_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/COXt_GfXa2M_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `116.334` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.660 |
| save_clips | - |
| sample_frames | 0.617 |
| caption_frames | 25.637 |
| sample_fps | 1.816 |
| detect_object_yolo | 7.202 |
| audio_scan | 8.569 |
| asr_timings | 12.046 |
| ast_timings | 18.047 |
| describe_scenes | 11.376 |
| summarize_scenes | 14.072 |
| synthesize_synopsis | 12.357 |
| make_embedding | 2.503 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 26.260 |
| branch_yolo_total | 9.024 |
| branch_audio_total | 38.672 |
