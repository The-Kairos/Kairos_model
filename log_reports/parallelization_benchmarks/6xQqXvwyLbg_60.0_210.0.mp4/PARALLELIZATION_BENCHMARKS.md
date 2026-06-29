# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 13:22:37 UTC | 6xQqXvwyLbg_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 240.256 | 0.781 | 92.457 | 26.041 | 43.423 | 16.373 | 3.963 |

## 2026-06-24 13:22:37 UTC | 6xQqXvwyLbg_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/6xQqXvwyLbg_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `240.256` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.781 |
| save_clips | - |
| sample_frames | 1.494 |
| caption_frames | 42.906 |
| sample_fps | 2.430 |
| detect_object_yolo | 9.008 |
| audio_scan | 11.780 |
| asr_timings | 48.332 |
| ast_timings | 32.336 |
| describe_scenes | 26.041 |
| summarize_scenes | 43.423 |
| synthesize_synopsis | 16.373 |
| make_embedding | 3.963 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.406 |
| branch_yolo_total | 11.443 |
| branch_audio_total | 92.457 |
