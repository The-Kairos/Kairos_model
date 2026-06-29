# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 15:30:34 UTC | QrFLjLZIeig_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 168.426 | 0.800 | 57.607 | 16.767 | 13.191 | 21.305 | 3.607 |

## 2026-06-25 15:30:34 UTC | QrFLjLZIeig_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/QrFLjLZIeig_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `168.426` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.800 |
| save_clips | - |
| sample_frames | 1.251 |
| caption_frames | 41.358 |
| sample_fps | 2.400 |
| detect_object_yolo | 8.723 |
| audio_scan | 16.601 |
| asr_timings | 11.448 |
| ast_timings | 29.551 |
| describe_scenes | 16.767 |
| summarize_scenes | 13.191 |
| synthesize_synopsis | 21.305 |
| make_embedding | 3.607 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 42.614 |
| branch_yolo_total | 11.128 |
| branch_audio_total | 57.607 |
