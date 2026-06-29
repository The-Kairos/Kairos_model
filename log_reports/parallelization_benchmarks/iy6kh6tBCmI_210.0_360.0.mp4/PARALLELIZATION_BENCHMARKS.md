# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 09:44:05 UTC | iy6kh6tBCmI_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 172.721 | 0.786 | 42.841 | 18.166 | 38.419 | 26.560 | 2.785 |

## 2026-06-26 09:44:05 UTC | iy6kh6tBCmI_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/iy6kh6tBCmI_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `172.721` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.786 |
| save_clips | - |
| sample_frames | 0.768 |
| caption_frames | 31.063 |
| sample_fps | 2.104 |
| detect_object_yolo | 7.770 |
| audio_scan | 9.772 |
| asr_timings | 11.701 |
| ast_timings | 21.360 |
| describe_scenes | 18.166 |
| summarize_scenes | 38.419 |
| synthesize_synopsis | 26.560 |
| make_embedding | 2.785 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 31.836 |
| branch_yolo_total | 9.880 |
| branch_audio_total | 42.841 |
