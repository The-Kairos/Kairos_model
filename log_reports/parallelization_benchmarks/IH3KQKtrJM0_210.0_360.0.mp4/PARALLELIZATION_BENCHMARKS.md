# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 04:39:01 UTC | IH3KQKtrJM0_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 137.184 | 0.773 | 49.337 | 13.675 | 12.691 | 11.432 | 3.111 |

## 2026-06-25 04:39:01 UTC | IH3KQKtrJM0_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/IH3KQKtrJM0_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `137.184` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.773 |
| save_clips | - |
| sample_frames | 0.933 |
| caption_frames | 34.008 |
| sample_fps | 2.139 |
| detect_object_yolo | 7.683 |
| audio_scan | 15.984 |
| asr_timings | 9.810 |
| ast_timings | 23.534 |
| describe_scenes | 13.675 |
| summarize_scenes | 12.691 |
| synthesize_synopsis | 11.432 |
| make_embedding | 3.111 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 34.947 |
| branch_yolo_total | 9.828 |
| branch_audio_total | 49.337 |
