# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 00:20:32 UTC | u-UA8t2EVpA_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 221.408 | 0.646 | 78.733 | 20.133 | 11.577 | 9.658 | 6.785 |

## 2026-06-27 00:20:32 UTC | u-UA8t2EVpA_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/u-UA8t2EVpA_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `221.408` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.646 |
| save_clips | - |
| sample_frames | 1.832 |
| caption_frames | 74.776 |
| sample_fps | 2.506 |
| detect_object_yolo | 13.297 |
| audio_scan | 10.695 |
| asr_timings | 13.090 |
| ast_timings | 54.940 |
| describe_scenes | 20.133 |
| summarize_scenes | 11.577 |
| synthesize_synopsis | 9.658 |
| make_embedding | 6.785 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 76.614 |
| branch_yolo_total | 15.809 |
| branch_audio_total | 78.733 |
