# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 06:02:41 UTC | hA1nRRHZ8tg_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 156.335 | 0.791 | 52.567 | 14.036 | 12.910 | 21.195 | 3.565 |

## 2026-06-26 06:02:41 UTC | hA1nRRHZ8tg_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/hA1nRRHZ8tg_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `156.335` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.791 |
| save_clips | - |
| sample_frames | 1.104 |
| caption_frames | 37.741 |
| sample_fps | 2.285 |
| detect_object_yolo | 8.730 |
| audio_scan | 13.962 |
| asr_timings | 9.020 |
| ast_timings | 29.577 |
| describe_scenes | 14.036 |
| summarize_scenes | 12.910 |
| synthesize_synopsis | 21.195 |
| make_embedding | 3.565 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.851 |
| branch_yolo_total | 11.021 |
| branch_audio_total | 52.567 |
