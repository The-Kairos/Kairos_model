# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 03:08:16 UTC | H7jdqCKg9x8_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 116.328 | 0.805 | 45.454 | 11.324 | 11.492 | 7.196 | 2.556 |

## 2026-06-25 03:08:16 UTC | H7jdqCKg9x8_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/H7jdqCKg9x8_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `116.328` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.805 |
| save_clips | - |
| sample_frames | 0.715 |
| caption_frames | 26.375 |
| sample_fps | 2.012 |
| detect_object_yolo | 7.016 |
| audio_scan | 14.931 |
| asr_timings | 12.095 |
| ast_timings | 18.419 |
| describe_scenes | 11.324 |
| summarize_scenes | 11.492 |
| synthesize_synopsis | 7.196 |
| make_embedding | 2.556 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 27.096 |
| branch_yolo_total | 9.033 |
| branch_audio_total | 45.454 |
