# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 22:13:21 UTC | Z3F41bP1Lcs_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 58.356 | 0.666 | 15.352 | 5.693 | 4.411 | 6.386 | 1.854 |

## 2026-06-25 22:13:21 UTC | Z3F41bP1Lcs_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Z3F41bP1Lcs_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `58.356` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.666 |
| save_clips | - |
| sample_frames | 0.322 |
| caption_frames | 15.024 |
| sample_fps | 1.711 |
| detect_object_yolo | 6.439 |
| audio_scan | 3.827 |
| asr_timings | 0.000 |
| ast_timings | 10.614 |
| describe_scenes | 5.693 |
| summarize_scenes | 4.411 |
| synthesize_synopsis | 6.386 |
| make_embedding | 1.854 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 15.352 |
| branch_yolo_total | 8.156 |
| branch_audio_total | 14.449 |
