# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 20:20:35 UTC | Vu0Z5BdPKaY_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 207.278 | 0.822 | 70.994 | 23.811 | 11.933 | 18.335 | 5.383 |

## 2026-06-25 20:20:35 UTC | Vu0Z5BdPKaY_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Vu0Z5BdPKaY_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `207.278` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.822 |
| save_clips | - |
| sample_frames | 1.268 |
| caption_frames | 59.556 |
| sample_fps | 2.424 |
| detect_object_yolo | 11.342 |
| audio_scan | 15.953 |
| asr_timings | 11.978 |
| ast_timings | 43.054 |
| describe_scenes | 23.811 |
| summarize_scenes | 11.933 |
| synthesize_synopsis | 18.335 |
| make_embedding | 5.383 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 60.829 |
| branch_yolo_total | 13.772 |
| branch_audio_total | 70.994 |
