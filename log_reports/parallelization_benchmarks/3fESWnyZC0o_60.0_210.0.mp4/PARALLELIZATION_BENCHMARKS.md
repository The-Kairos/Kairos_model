# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 22:19:49 UTC | 3fESWnyZC0o_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 188.712 | 0.799 | 69.594 | 13.028 | 10.433 | 8.806 | 5.705 |

## 2026-06-21 22:19:49 UTC | 3fESWnyZC0o_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3fESWnyZC0o_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `188.712` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.799 |
| save_clips | - |
| sample_frames | 1.880 |
| caption_frames | 61.991 |
| sample_fps | 2.676 |
| detect_object_yolo | 12.332 |
| audio_scan | 12.981 |
| asr_timings | 10.794 |
| ast_timings | 45.810 |
| describe_scenes | 13.028 |
| summarize_scenes | 10.433 |
| synthesize_synopsis | 8.806 |
| make_embedding | 5.705 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 63.877 |
| branch_yolo_total | 15.013 |
| branch_audio_total | 69.594 |
