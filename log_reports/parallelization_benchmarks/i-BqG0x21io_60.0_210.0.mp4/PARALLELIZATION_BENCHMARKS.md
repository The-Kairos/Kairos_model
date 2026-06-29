# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 07:41:07 UTC | i-BqG0x21io_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 232.803 | 0.793 | 74.480 | 30.995 | 20.356 | 23.744 | 5.424 |

## 2026-06-26 07:41:07 UTC | i-BqG0x21io_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/i-BqG0x21io_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `232.803` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.793 |
| save_clips | - |
| sample_frames | 1.367 |
| caption_frames | 60.142 |
| sample_fps | 2.563 |
| detect_object_yolo | 11.440 |
| audio_scan | 16.390 |
| asr_timings | 13.413 |
| ast_timings | 44.669 |
| describe_scenes | 30.995 |
| summarize_scenes | 20.356 |
| synthesize_synopsis | 23.744 |
| make_embedding | 5.424 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 61.515 |
| branch_yolo_total | 14.009 |
| branch_audio_total | 74.480 |
