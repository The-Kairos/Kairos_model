# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 07:22:17 UTC | hxdrPdD8nnA_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 192.957 | 0.804 | 62.403 | 23.765 | 16.775 | 20.092 | 4.539 |

## 2026-06-26 07:22:17 UTC | hxdrPdD8nnA_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/hxdrPdD8nnA_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `192.957` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.804 |
| save_clips | - |
| sample_frames | 1.142 |
| caption_frames | 49.412 |
| sample_fps | 2.338 |
| detect_object_yolo | 10.265 |
| audio_scan | 13.948 |
| asr_timings | 10.233 |
| ast_timings | 38.213 |
| describe_scenes | 23.765 |
| summarize_scenes | 16.775 |
| synthesize_synopsis | 20.092 |
| make_embedding | 4.539 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 50.560 |
| branch_yolo_total | 12.609 |
| branch_audio_total | 62.403 |
