# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 19:34:21 UTC | s7MesIm6VLw_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 243.297 | 0.648 | 98.633 | 28.285 | 18.582 | 15.517 | 5.426 |

## 2026-06-26 19:34:21 UTC | s7MesIm6VLw_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/s7MesIm6VLw_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `243.297` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.648 |
| save_clips | - |
| sample_frames | 1.371 |
| caption_frames | 59.175 |
| sample_fps | 2.237 |
| detect_object_yolo | 11.923 |
| audio_scan | 13.069 |
| asr_timings | 41.490 |
| ast_timings | 44.066 |
| describe_scenes | 28.285 |
| summarize_scenes | 18.582 |
| synthesize_synopsis | 15.517 |
| make_embedding | 5.426 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 60.552 |
| branch_yolo_total | 14.166 |
| branch_audio_total | 98.633 |
