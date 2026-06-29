# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 07:14:38 UTC | LPLJ1mPAs7M_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 185.368 | 0.624 | 53.652 | 30.010 | 13.067 | 22.014 | 4.207 |

## 2026-06-25 07:14:38 UTC | LPLJ1mPAs7M_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/LPLJ1mPAs7M_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `185.368` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.624 |
| save_clips | - |
| sample_frames | 1.094 |
| caption_frames | 47.385 |
| sample_fps | 2.160 |
| detect_object_yolo | 9.703 |
| audio_scan | 8.564 |
| asr_timings | 9.973 |
| ast_timings | 35.106 |
| describe_scenes | 30.010 |
| summarize_scenes | 13.067 |
| synthesize_synopsis | 22.014 |
| make_embedding | 4.207 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 48.485 |
| branch_yolo_total | 11.869 |
| branch_audio_total | 53.652 |
