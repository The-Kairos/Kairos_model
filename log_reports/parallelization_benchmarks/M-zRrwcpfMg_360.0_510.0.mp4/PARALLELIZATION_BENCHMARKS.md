# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 08:04:06 UTC | M-zRrwcpfMg_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 199.364 | 0.797 | 72.215 | 19.894 | 19.791 | 23.718 | 3.896 |

## 2026-06-25 08:04:06 UTC | M-zRrwcpfMg_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/M-zRrwcpfMg_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `199.364` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.797 |
| save_clips | - |
| sample_frames | 1.309 |
| caption_frames | 44.794 |
| sample_fps | 2.390 |
| detect_object_yolo | 9.128 |
| audio_scan | 14.855 |
| asr_timings | 26.293 |
| ast_timings | 31.058 |
| describe_scenes | 19.894 |
| summarize_scenes | 19.791 |
| synthesize_synopsis | 23.718 |
| make_embedding | 3.896 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.109 |
| branch_yolo_total | 11.524 |
| branch_audio_total | 72.215 |
