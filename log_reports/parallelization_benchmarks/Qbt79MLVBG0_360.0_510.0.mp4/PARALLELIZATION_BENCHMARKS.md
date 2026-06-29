# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 15:12:32 UTC | Qbt79MLVBG0_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 102.754 | 1.528 | 28.576 | 9.991 | 5.423 | 38.244 | 1.385 |

## 2026-06-25 15:12:32 UTC | Qbt79MLVBG0_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Qbt79MLVBG0_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `102.754` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.528 |
| save_clips | - |
| sample_frames | 0.148 |
| caption_frames | 8.512 |
| sample_fps | 1.788 |
| detect_object_yolo | 5.759 |
| audio_scan | 13.482 |
| asr_timings | 10.603 |
| ast_timings | 4.483 |
| describe_scenes | 9.991 |
| summarize_scenes | 5.423 |
| synthesize_synopsis | 38.244 |
| make_embedding | 1.385 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 8.666 |
| branch_yolo_total | 7.553 |
| branch_audio_total | 28.576 |
