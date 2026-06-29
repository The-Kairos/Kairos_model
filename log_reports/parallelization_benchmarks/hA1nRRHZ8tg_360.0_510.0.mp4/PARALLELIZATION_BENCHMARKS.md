# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 06:05:45 UTC | hA1nRRHZ8tg_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 183.231 | 0.780 | 57.653 | 12.035 | 16.929 | 33.670 | 3.959 |

## 2026-06-26 06:05:45 UTC | hA1nRRHZ8tg_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/hA1nRRHZ8tg_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `183.231` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.780 |
| save_clips | - |
| sample_frames | 1.199 |
| caption_frames | 43.614 |
| sample_fps | 2.355 |
| detect_object_yolo | 9.629 |
| audio_scan | 14.997 |
| asr_timings | 9.793 |
| ast_timings | 32.854 |
| describe_scenes | 12.035 |
| summarize_scenes | 16.929 |
| synthesize_synopsis | 33.670 |
| make_embedding | 3.959 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.819 |
| branch_yolo_total | 11.990 |
| branch_audio_total | 57.653 |
