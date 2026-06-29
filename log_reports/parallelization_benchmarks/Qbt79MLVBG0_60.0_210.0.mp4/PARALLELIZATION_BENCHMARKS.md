# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 15:14:34 UTC | Qbt79MLVBG0_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 120.626 | 1.584 | 30.137 | 18.356 | 22.435 | 16.278 | 2.127 |

## 2026-06-25 15:14:34 UTC | Qbt79MLVBG0_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Qbt79MLVBG0_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `120.626` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.584 |
| save_clips | - |
| sample_frames | 0.490 |
| caption_frames | 19.283 |
| sample_fps | 1.921 |
| detect_object_yolo | 6.625 |
| audio_scan | 7.663 |
| asr_timings | 9.479 |
| ast_timings | 12.987 |
| describe_scenes | 18.356 |
| summarize_scenes | 22.435 |
| synthesize_synopsis | 16.278 |
| make_embedding | 2.127 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 19.779 |
| branch_yolo_total | 8.552 |
| branch_audio_total | 30.137 |
