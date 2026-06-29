# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 23:53:42 UTC | EoojNzIL2cQ_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 154.602 | 0.791 | 53.253 | 9.438 | 16.363 | 17.497 | 3.558 |

## 2026-06-24 23:53:42 UTC | EoojNzIL2cQ_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/EoojNzIL2cQ_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `154.602` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.791 |
| save_clips | - |
| sample_frames | 0.981 |
| caption_frames | 40.081 |
| sample_fps | 2.271 |
| detect_object_yolo | 8.972 |
| audio_scan | 12.868 |
| asr_timings | 10.767 |
| ast_timings | 29.609 |
| describe_scenes | 9.438 |
| summarize_scenes | 16.363 |
| synthesize_synopsis | 17.497 |
| make_embedding | 3.558 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.068 |
| branch_yolo_total | 11.248 |
| branch_audio_total | 53.253 |
