# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 19:14:07 UTC | s6yCM9pAQ0o_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 167.734 | 0.646 | 69.341 | 12.710 | 6.351 | 16.236 | 3.839 |

## 2026-06-26 19:14:07 UTC | s6yCM9pAQ0o_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/s6yCM9pAQ0o_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `167.734` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.646 |
| save_clips | - |
| sample_frames | 1.037 |
| caption_frames | 44.391 |
| sample_fps | 2.066 |
| detect_object_yolo | 9.576 |
| audio_scan | 14.857 |
| asr_timings | 22.010 |
| ast_timings | 32.465 |
| describe_scenes | 12.710 |
| summarize_scenes | 6.351 |
| synthesize_synopsis | 16.236 |
| make_embedding | 3.839 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.434 |
| branch_yolo_total | 11.649 |
| branch_audio_total | 69.341 |
