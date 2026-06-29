# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 22:12:21 UTC | Z2eyocRl2mo_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 115.156 | 0.639 | 38.952 | 8.945 | 16.454 | 10.611 | 2.488 |

## 2026-06-25 22:12:21 UTC | Z2eyocRl2mo_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Z2eyocRl2mo_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `115.156` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.639 |
| save_clips | - |
| sample_frames | 0.601 |
| caption_frames | 25.841 |
| sample_fps | 1.906 |
| detect_object_yolo | 7.317 |
| audio_scan | 8.640 |
| asr_timings | 11.481 |
| ast_timings | 18.822 |
| describe_scenes | 8.945 |
| summarize_scenes | 16.454 |
| synthesize_synopsis | 10.611 |
| make_embedding | 2.488 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 26.448 |
| branch_yolo_total | 9.229 |
| branch_audio_total | 38.952 |
