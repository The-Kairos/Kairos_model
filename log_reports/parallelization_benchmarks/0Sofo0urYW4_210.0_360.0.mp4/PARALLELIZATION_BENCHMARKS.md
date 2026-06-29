# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 13:38:26 UTC | 0Sofo0urYW4_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 168.372 | 0.793 | 58.323 | 13.267 | 20.156 | 7.011 | 4.504 |

## 2026-06-27 13:38:26 UTC | 0Sofo0urYW4_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0Sofo0urYW4_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `168.372` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.793 |
| save_clips | - |
| sample_frames | 1.305 |
| caption_frames | 48.988 |
| sample_fps | 2.384 |
| detect_object_yolo | 10.244 |
| audio_scan | 9.649 |
| asr_timings | 10.945 |
| ast_timings | 37.721 |
| describe_scenes | 13.267 |
| summarize_scenes | 20.156 |
| synthesize_synopsis | 7.011 |
| make_embedding | 4.504 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 50.299 |
| branch_yolo_total | 12.634 |
| branch_audio_total | 58.323 |
