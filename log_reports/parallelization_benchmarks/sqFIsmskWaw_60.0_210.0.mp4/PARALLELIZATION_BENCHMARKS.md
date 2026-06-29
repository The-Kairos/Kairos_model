# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 22:40:42 UTC | sqFIsmskWaw_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 144.170 | 0.786 | 51.629 | 11.624 | 11.865 | 9.801 | 3.533 |

## 2026-06-26 22:40:42 UTC | sqFIsmskWaw_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/sqFIsmskWaw_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `144.170` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.786 |
| save_clips | - |
| sample_frames | 1.176 |
| caption_frames | 40.923 |
| sample_fps | 2.298 |
| detect_object_yolo | 9.122 |
| audio_scan | 13.744 |
| asr_timings | 7.379 |
| ast_timings | 30.498 |
| describe_scenes | 11.624 |
| summarize_scenes | 11.865 |
| synthesize_synopsis | 9.801 |
| make_embedding | 3.533 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 42.105 |
| branch_yolo_total | 11.426 |
| branch_audio_total | 51.629 |
