# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 07:22:04 UTC | LPLJ1mPAs7M_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 254.398 | 0.624 | 59.683 | 25.771 | 62.765 | 26.466 | 5.155 |

## 2026-06-25 07:22:04 UTC | LPLJ1mPAs7M_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/LPLJ1mPAs7M_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `254.398` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.624 |
| save_clips | - |
| sample_frames | 1.300 |
| caption_frames | 57.897 |
| sample_fps | 2.343 |
| detect_object_yolo | 10.930 |
| audio_scan | 8.629 |
| asr_timings | 10.505 |
| ast_timings | 40.540 |
| describe_scenes | 25.771 |
| summarize_scenes | 62.765 |
| synthesize_synopsis | 26.466 |
| make_embedding | 5.155 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 59.203 |
| branch_yolo_total | 13.279 |
| branch_audio_total | 59.683 |
