# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 20:01:49 UTC | BwzSUld6bS8_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 131.977 | 0.656 | 48.436 | 11.696 | 9.179 | 15.251 | 3.118 |

## 2026-06-24 20:01:49 UTC | BwzSUld6bS8_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/BwzSUld6bS8_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `131.977` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.656 |
| save_clips | - |
| sample_frames | 0.765 |
| caption_frames | 31.507 |
| sample_fps | 1.963 |
| detect_object_yolo | 8.026 |
| audio_scan | 15.018 |
| asr_timings | 10.007 |
| ast_timings | 23.403 |
| describe_scenes | 11.696 |
| summarize_scenes | 9.179 |
| synthesize_synopsis | 15.251 |
| make_embedding | 3.118 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.278 |
| branch_yolo_total | 9.994 |
| branch_audio_total | 48.436 |
