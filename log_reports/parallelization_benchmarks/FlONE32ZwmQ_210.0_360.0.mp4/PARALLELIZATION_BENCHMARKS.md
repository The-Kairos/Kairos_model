# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 00:23:39 UTC | FlONE32ZwmQ_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 134.647 | 0.632 | 46.071 | 8.506 | 21.808 | 9.077 | 3.107 |

## 2026-06-25 00:23:39 UTC | FlONE32ZwmQ_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/FlONE32ZwmQ_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `134.647` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.632 |
| save_clips | - |
| sample_frames | 0.829 |
| caption_frames | 33.026 |
| sample_fps | 1.991 |
| detect_object_yolo | 8.210 |
| audio_scan | 10.737 |
| asr_timings | 10.790 |
| ast_timings | 24.536 |
| describe_scenes | 8.506 |
| summarize_scenes | 21.808 |
| synthesize_synopsis | 9.077 |
| make_embedding | 3.107 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 33.860 |
| branch_yolo_total | 10.206 |
| branch_audio_total | 46.071 |
