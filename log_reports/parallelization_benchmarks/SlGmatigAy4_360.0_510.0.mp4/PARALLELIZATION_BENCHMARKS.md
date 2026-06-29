# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 17:38:29 UTC | SlGmatigAy4_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 193.707 | 0.701 | 60.195 | 15.551 | 37.501 | 14.812 | 4.166 |

## 2026-06-25 17:38:29 UTC | SlGmatigAy4_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/SlGmatigAy4_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `193.707` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.701 |
| save_clips | - |
| sample_frames | 1.267 |
| caption_frames | 46.736 |
| sample_fps | 2.274 |
| detect_object_yolo | 9.112 |
| audio_scan | 16.138 |
| asr_timings | 8.909 |
| ast_timings | 35.139 |
| describe_scenes | 15.551 |
| summarize_scenes | 37.501 |
| synthesize_synopsis | 14.812 |
| make_embedding | 4.166 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 48.009 |
| branch_yolo_total | 11.392 |
| branch_audio_total | 60.195 |
