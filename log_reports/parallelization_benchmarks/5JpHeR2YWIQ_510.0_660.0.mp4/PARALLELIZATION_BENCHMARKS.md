# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 11:47:45 UTC | 5JpHeR2YWIQ_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 168.113 | 0.808 | 54.517 | 17.138 | 9.519 | 26.000 | 3.593 |

## 2026-06-24 11:47:45 UTC | 5JpHeR2YWIQ_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/5JpHeR2YWIQ_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `168.113` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.808 |
| save_clips | - |
| sample_frames | 1.312 |
| caption_frames | 42.393 |
| sample_fps | 2.389 |
| detect_object_yolo | 9.039 |
| audio_scan | 11.788 |
| asr_timings | 13.171 |
| ast_timings | 29.549 |
| describe_scenes | 17.138 |
| summarize_scenes | 9.519 |
| synthesize_synopsis | 26.000 |
| make_embedding | 3.593 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.711 |
| branch_yolo_total | 11.433 |
| branch_audio_total | 54.517 |
