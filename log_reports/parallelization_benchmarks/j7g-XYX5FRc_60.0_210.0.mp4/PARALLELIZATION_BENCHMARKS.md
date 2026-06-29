# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 10:15:34 UTC | j7g-XYX5FRc_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 228.303 | 0.795 | 71.344 | 25.534 | 18.789 | 28.034 | 5.511 |

## 2026-06-26 10:15:34 UTC | j7g-XYX5FRc_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/j7g-XYX5FRc_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `228.303` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.795 |
| save_clips | - |
| sample_frames | 1.539 |
| caption_frames | 61.686 |
| sample_fps | 2.525 |
| detect_object_yolo | 11.115 |
| audio_scan | 16.152 |
| asr_timings | 10.663 |
| ast_timings | 44.521 |
| describe_scenes | 25.534 |
| summarize_scenes | 18.789 |
| synthesize_synopsis | 28.034 |
| make_embedding | 5.511 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 63.231 |
| branch_yolo_total | 13.646 |
| branch_audio_total | 71.344 |
