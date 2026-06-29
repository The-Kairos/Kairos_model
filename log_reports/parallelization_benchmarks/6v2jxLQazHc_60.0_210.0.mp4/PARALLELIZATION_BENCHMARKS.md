# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 13:12:07 UTC | 6v2jxLQazHc_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 203.974 | 0.681 | 59.170 | 20.950 | 22.335 | 43.226 | 3.883 |

## 2026-06-24 13:12:07 UTC | 6v2jxLQazHc_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/6v2jxLQazHc_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `203.974` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.681 |
| save_clips | - |
| sample_frames | 0.910 |
| caption_frames | 40.280 |
| sample_fps | 2.103 |
| detect_object_yolo | 9.045 |
| audio_scan | 9.655 |
| asr_timings | 17.074 |
| ast_timings | 32.433 |
| describe_scenes | 20.950 |
| summarize_scenes | 22.335 |
| synthesize_synopsis | 43.226 |
| make_embedding | 3.883 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.196 |
| branch_yolo_total | 11.154 |
| branch_audio_total | 59.170 |
