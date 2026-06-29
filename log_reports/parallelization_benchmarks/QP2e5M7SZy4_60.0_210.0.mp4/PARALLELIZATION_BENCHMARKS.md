# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 15:08:51 UTC | QP2e5M7SZy4_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 197.443 | 0.782 | 57.301 | 31.568 | 26.391 | 21.125 | 3.848 |

## 2026-06-25 15:08:51 UTC | QP2e5M7SZy4_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/QP2e5M7SZy4_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `197.443` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.782 |
| save_clips | - |
| sample_frames | 1.035 |
| caption_frames | 42.375 |
| sample_fps | 2.274 |
| detect_object_yolo | 9.335 |
| audio_scan | 12.334 |
| asr_timings | 13.441 |
| ast_timings | 31.518 |
| describe_scenes | 31.568 |
| summarize_scenes | 26.391 |
| synthesize_synopsis | 21.125 |
| make_embedding | 3.848 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.416 |
| branch_yolo_total | 11.615 |
| branch_audio_total | 57.301 |
