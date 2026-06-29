# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 13:44:02 UTC | 0Sofo0urYW4_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 156.664 | 0.811 | 54.591 | 12.583 | 14.801 | 10.862 | 3.853 |

## 2026-06-27 13:44:02 UTC | 0Sofo0urYW4_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0Sofo0urYW4_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `156.664` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.811 |
| save_clips | - |
| sample_frames | 1.267 |
| caption_frames | 44.248 |
| sample_fps | 2.374 |
| detect_object_yolo | 9.875 |
| audio_scan | 11.759 |
| asr_timings | 9.995 |
| ast_timings | 32.828 |
| describe_scenes | 12.583 |
| summarize_scenes | 14.801 |
| synthesize_synopsis | 10.862 |
| make_embedding | 3.853 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.522 |
| branch_yolo_total | 12.255 |
| branch_audio_total | 54.591 |
