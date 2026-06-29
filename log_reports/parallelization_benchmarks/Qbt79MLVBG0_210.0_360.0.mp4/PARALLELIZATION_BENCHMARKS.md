# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 15:10:49 UTC | Qbt79MLVBG0_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 116.038 | 0.775 | 31.828 | 9.323 | 9.041 | 37.359 | 1.848 |

## 2026-06-25 15:10:49 UTC | Qbt79MLVBG0_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Qbt79MLVBG0_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `116.038` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.775 |
| save_clips | - |
| sample_frames | 0.372 |
| caption_frames | 15.631 |
| sample_fps | 1.894 |
| detect_object_yolo | 6.542 |
| audio_scan | 12.389 |
| asr_timings | 9.442 |
| ast_timings | 9.988 |
| describe_scenes | 9.323 |
| summarize_scenes | 9.041 |
| synthesize_synopsis | 37.359 |
| make_embedding | 1.848 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 16.009 |
| branch_yolo_total | 8.441 |
| branch_audio_total | 31.828 |
