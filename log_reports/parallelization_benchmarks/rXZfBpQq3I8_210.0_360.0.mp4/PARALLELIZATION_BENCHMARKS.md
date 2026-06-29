# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 18:45:46 UTC | rXZfBpQq3I8_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 245.875 | 0.809 | 79.317 | 26.979 | 29.734 | 10.698 | 6.719 |

## 2026-06-26 18:45:46 UTC | rXZfBpQq3I8_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/rXZfBpQq3I8_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `245.875` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.809 |
| save_clips | - |
| sample_frames | 1.656 |
| caption_frames | 72.281 |
| sample_fps | 2.623 |
| detect_object_yolo | 13.614 |
| audio_scan | 15.074 |
| asr_timings | 11.112 |
| ast_timings | 53.123 |
| describe_scenes | 26.979 |
| summarize_scenes | 29.734 |
| synthesize_synopsis | 10.698 |
| make_embedding | 6.719 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 73.943 |
| branch_yolo_total | 16.243 |
| branch_audio_total | 79.317 |
