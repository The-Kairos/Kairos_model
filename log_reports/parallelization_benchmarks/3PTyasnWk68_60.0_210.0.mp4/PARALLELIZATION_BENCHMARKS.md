# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 22:04:44 UTC | 3PTyasnWk68_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 144.486 | 0.658 | 57.369 | 9.347 | 7.824 | 6.347 | 3.901 |

## 2026-06-21 22:04:44 UTC | 3PTyasnWk68_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3PTyasnWk68_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `144.486` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.658 |
| save_clips | - |
| sample_frames | 1.162 |
| caption_frames | 44.487 |
| sample_fps | 2.170 |
| detect_object_yolo | 9.761 |
| audio_scan | 15.015 |
| asr_timings | 9.234 |
| ast_timings | 33.111 |
| describe_scenes | 9.347 |
| summarize_scenes | 7.824 |
| synthesize_synopsis | 6.347 |
| make_embedding | 3.901 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.655 |
| branch_yolo_total | 11.937 |
| branch_audio_total | 57.369 |
