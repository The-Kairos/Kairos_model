# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 23:18:44 UTC | Dz0MY6ARnU4_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 142.204 | 0.676 | 86.621 | 5.101 | 4.673 | 8.802 | 2.253 |

## 2026-06-24 23:18:44 UTC | Dz0MY6ARnU4_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Dz0MY6ARnU4_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `142.204` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.676 |
| save_clips | - |
| sample_frames | 0.632 |
| caption_frames | 22.527 |
| sample_fps | 1.875 |
| detect_object_yolo | 7.641 |
| audio_scan | 15.992 |
| asr_timings | 54.572 |
| ast_timings | 16.048 |
| describe_scenes | 5.101 |
| summarize_scenes | 4.673 |
| synthesize_synopsis | 8.802 |
| make_embedding | 2.253 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 23.164 |
| branch_yolo_total | 9.521 |
| branch_audio_total | 86.621 |
