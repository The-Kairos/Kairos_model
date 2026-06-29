# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 07:47:28 UTC | i0dywTgTRy4_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 126.769 | 0.667 | 39.297 | 12.101 | 11.730 | 22.822 | 2.604 |

## 2026-06-26 07:47:28 UTC | i0dywTgTRy4_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/i0dywTgTRy4_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `126.769` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.667 |
| save_clips | - |
| sample_frames | 0.614 |
| caption_frames | 26.303 |
| sample_fps | 1.914 |
| detect_object_yolo | 7.302 |
| audio_scan | 11.881 |
| asr_timings | 9.360 |
| ast_timings | 18.047 |
| describe_scenes | 12.101 |
| summarize_scenes | 11.730 |
| synthesize_synopsis | 22.822 |
| make_embedding | 2.604 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 26.924 |
| branch_yolo_total | 9.221 |
| branch_audio_total | 39.297 |
