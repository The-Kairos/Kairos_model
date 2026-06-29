# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 16:55:03 UTC | S8ZUWshc1C4_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 148.459 | 0.651 | 50.157 | 17.640 | 11.721 | 16.754 | 3.667 |

## 2026-06-25 16:55:03 UTC | S8ZUWshc1C4_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/S8ZUWshc1C4_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `148.459` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.651 |
| save_clips | - |
| sample_frames | 1.047 |
| caption_frames | 34.128 |
| sample_fps | 2.100 |
| detect_object_yolo | 9.111 |
| audio_scan | 13.847 |
| asr_timings | 9.597 |
| ast_timings | 26.704 |
| describe_scenes | 17.640 |
| summarize_scenes | 11.721 |
| synthesize_synopsis | 16.754 |
| make_embedding | 3.667 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 35.181 |
| branch_yolo_total | 11.217 |
| branch_audio_total | 50.157 |
