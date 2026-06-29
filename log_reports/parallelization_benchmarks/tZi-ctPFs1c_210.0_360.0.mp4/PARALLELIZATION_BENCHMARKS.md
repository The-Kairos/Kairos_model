# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 23:40:15 UTC | tZi-ctPFs1c_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 159.786 | 0.773 | 58.567 | 14.118 | 9.567 | 10.769 | 4.105 |

## 2026-06-26 23:40:15 UTC | tZi-ctPFs1c_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/tZi-ctPFs1c_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `159.786` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.773 |
| save_clips | - |
| sample_frames | 1.538 |
| caption_frames | 46.674 |
| sample_fps | 2.458 |
| detect_object_yolo | 9.797 |
| audio_scan | 13.868 |
| asr_timings | 8.593 |
| ast_timings | 36.097 |
| describe_scenes | 14.118 |
| summarize_scenes | 9.567 |
| synthesize_synopsis | 10.769 |
| make_embedding | 4.105 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 48.218 |
| branch_yolo_total | 12.260 |
| branch_audio_total | 58.567 |
