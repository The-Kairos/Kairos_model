# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 18:17:36 UTC | mQxsInE7UQ4_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 226.119 | 0.778 | 69.616 | 29.340 | 24.378 | 20.012 | 5.456 |

## 2026-06-26 18:17:36 UTC | mQxsInE7UQ4_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/mQxsInE7UQ4_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `226.119` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.778 |
| save_clips | - |
| sample_frames | 1.348 |
| caption_frames | 59.600 |
| sample_fps | 2.519 |
| detect_object_yolo | 11.647 |
| audio_scan | 15.967 |
| asr_timings | 10.255 |
| ast_timings | 43.385 |
| describe_scenes | 29.340 |
| summarize_scenes | 24.378 |
| synthesize_synopsis | 20.012 |
| make_embedding | 5.456 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 60.954 |
| branch_yolo_total | 14.171 |
| branch_audio_total | 69.616 |
