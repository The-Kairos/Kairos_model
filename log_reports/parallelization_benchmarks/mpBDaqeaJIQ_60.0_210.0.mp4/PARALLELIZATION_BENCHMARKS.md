# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 16:16:38 UTC | mpBDaqeaJIQ_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 165.508 | 0.771 | 60.787 | 10.984 | 8.652 | 10.035 | 5.490 |

## 2026-06-27 16:16:38 UTC | mpBDaqeaJIQ_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/mpBDaqeaJIQ_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `165.508` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.771 |
| save_clips | - |
| sample_frames | 1.435 |
| caption_frames | 53.057 |
| sample_fps | 2.382 |
| detect_object_yolo | 10.491 |
| audio_scan | 7.234 |
| asr_timings | 12.098 |
| ast_timings | 41.447 |
| describe_scenes | 10.984 |
| summarize_scenes | 8.652 |
| synthesize_synopsis | 10.035 |
| make_embedding | 5.490 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 54.498 |
| branch_yolo_total | 12.879 |
| branch_audio_total | 60.787 |
