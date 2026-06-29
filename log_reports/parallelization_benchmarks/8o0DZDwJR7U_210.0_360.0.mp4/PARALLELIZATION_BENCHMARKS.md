# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 17:04:30 UTC | 8o0DZDwJR7U_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 166.745 | 0.841 | 56.644 | 16.511 | 9.100 | 21.335 | 3.937 |

## 2026-06-24 17:04:30 UTC | 8o0DZDwJR7U_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/8o0DZDwJR7U_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `166.745` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.841 |
| save_clips | - |
| sample_frames | 1.487 |
| caption_frames | 43.413 |
| sample_fps | 2.490 |
| detect_object_yolo | 9.581 |
| audio_scan | 10.702 |
| asr_timings | 13.204 |
| ast_timings | 32.728 |
| describe_scenes | 16.511 |
| summarize_scenes | 9.100 |
| synthesize_synopsis | 21.335 |
| make_embedding | 3.937 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.905 |
| branch_yolo_total | 12.077 |
| branch_audio_total | 56.644 |
