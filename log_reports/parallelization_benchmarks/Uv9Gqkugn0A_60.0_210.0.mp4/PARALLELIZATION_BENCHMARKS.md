# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 19:35:53 UTC | Uv9Gqkugn0A_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 132.433 | 0.786 | 44.071 | 15.784 | 10.682 | 21.644 | 2.524 |

## 2026-06-25 19:35:53 UTC | Uv9Gqkugn0A_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Uv9Gqkugn0A_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `132.433` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.786 |
| save_clips | - |
| sample_frames | 0.735 |
| caption_frames | 25.585 |
| sample_fps | 2.054 |
| detect_object_yolo | 7.173 |
| audio_scan | 15.997 |
| asr_timings | 9.383 |
| ast_timings | 18.682 |
| describe_scenes | 15.784 |
| summarize_scenes | 10.682 |
| synthesize_synopsis | 21.644 |
| make_embedding | 2.524 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 26.325 |
| branch_yolo_total | 9.233 |
| branch_audio_total | 44.071 |
