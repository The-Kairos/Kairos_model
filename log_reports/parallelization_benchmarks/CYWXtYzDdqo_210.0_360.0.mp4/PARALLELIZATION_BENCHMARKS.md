# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 20:42:04 UTC | CYWXtYzDdqo_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 163.013 | 0.824 | 59.282 | 14.218 | 9.597 | 13.367 | 3.872 |

## 2026-06-24 20:42:04 UTC | CYWXtYzDdqo_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/CYWXtYzDdqo_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `163.013` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.824 |
| save_clips | - |
| sample_frames | 1.595 |
| caption_frames | 46.310 |
| sample_fps | 2.467 |
| detect_object_yolo | 10.032 |
| audio_scan | 11.786 |
| asr_timings | 14.422 |
| ast_timings | 33.065 |
| describe_scenes | 14.218 |
| summarize_scenes | 9.597 |
| synthesize_synopsis | 13.367 |
| make_embedding | 3.872 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.910 |
| branch_yolo_total | 12.505 |
| branch_audio_total | 59.282 |
