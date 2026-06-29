# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 00:20:44 UTC | aIGR9knS1B0_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 151.335 | 0.691 | 46.669 | 13.878 | 18.834 | 13.580 | 4.118 |

## 2026-06-26 00:20:44 UTC | aIGR9knS1B0_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/aIGR9knS1B0_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `151.335` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.691 |
| save_clips | - |
| sample_frames | 1.304 |
| caption_frames | 45.360 |
| sample_fps | 2.274 |
| detect_object_yolo | 9.430 |
| audio_scan | 3.831 |
| asr_timings | 0.000 |
| ast_timings | 36.630 |
| describe_scenes | 13.878 |
| summarize_scenes | 18.834 |
| synthesize_synopsis | 13.580 |
| make_embedding | 4.118 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.669 |
| branch_yolo_total | 11.710 |
| branch_audio_total | 40.469 |
