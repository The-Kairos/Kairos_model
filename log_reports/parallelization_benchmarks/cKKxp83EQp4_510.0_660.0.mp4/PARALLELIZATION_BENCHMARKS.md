# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 02:26:00 UTC | cKKxp83EQp4_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 85.718 | 0.790 | 22.876 | 8.686 | 8.523 | 12.359 | 2.279 |

## 2026-06-26 02:26:00 UTC | cKKxp83EQp4_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/cKKxp83EQp4_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `85.718` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.790 |
| save_clips | - |
| sample_frames | 0.554 |
| caption_frames | 22.317 |
| sample_fps | 1.988 |
| detect_object_yolo | 6.780 |
| audio_scan | 3.874 |
| asr_timings | 0.000 |
| ast_timings | 16.157 |
| describe_scenes | 8.686 |
| summarize_scenes | 8.523 |
| synthesize_synopsis | 12.359 |
| make_embedding | 2.279 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 22.876 |
| branch_yolo_total | 8.774 |
| branch_audio_total | 20.040 |
