# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 09:22:37 UTC | MOThH7E8fzc_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 1580.376 | 0.695 | 1473.385 | 18.793 | 18.660 | 22.333 | 2.825 |

## 2026-06-25 09:22:37 UTC | MOThH7E8fzc_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/MOThH7E8fzc_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `1580.376` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.695 |
| save_clips | - |
| sample_frames | 0.883 |
| caption_frames | 31.265 |
| sample_fps | 2.020 |
| detect_object_yolo | 8.085 |
| audio_scan | 8.585 |
| asr_timings | 1443.086 |
| ast_timings | 21.705 |
| describe_scenes | 18.793 |
| summarize_scenes | 18.660 |
| synthesize_synopsis | 22.333 |
| make_embedding | 2.825 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.153 |
| branch_yolo_total | 10.111 |
| branch_audio_total | 1473.385 |
