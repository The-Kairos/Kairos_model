# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 16:39:45 UTC | n_62dLVgd3I_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 81.469 | 0.658 | 43.546 | 4.737 | 1.888 | 9.550 | 1.535 |

## 2026-06-27 16:39:45 UTC | n_62dLVgd3I_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/n_62dLVgd3I_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `81.469` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.658 |
| save_clips | - |
| sample_frames | 0.210 |
| caption_frames | 10.376 |
| sample_fps | 1.675 |
| detect_object_yolo | 5.932 |
| audio_scan | 12.671 |
| asr_timings | 23.944 |
| ast_timings | 6.923 |
| describe_scenes | 4.737 |
| summarize_scenes | 1.888 |
| synthesize_synopsis | 9.550 |
| make_embedding | 1.535 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 10.593 |
| branch_yolo_total | 7.612 |
| branch_audio_total | 43.546 |
