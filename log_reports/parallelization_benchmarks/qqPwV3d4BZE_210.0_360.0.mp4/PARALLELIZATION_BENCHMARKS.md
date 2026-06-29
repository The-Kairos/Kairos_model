# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 09:05:13 UTC | qqPwV3d4BZE_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 134.143 | 0.789 | 50.257 | 8.785 | 8.094 | 9.629 | 3.525 |

## 2026-06-28 09:05:13 UTC | qqPwV3d4BZE_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/qqPwV3d4BZE_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `134.143` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.789 |
| save_clips | - |
| sample_frames | 1.143 |
| caption_frames | 39.667 |
| sample_fps | 2.315 |
| detect_object_yolo | 8.551 |
| audio_scan | 11.756 |
| asr_timings | 9.199 |
| ast_timings | 29.293 |
| describe_scenes | 8.785 |
| summarize_scenes | 8.094 |
| synthesize_synopsis | 9.629 |
| make_embedding | 3.525 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 40.816 |
| branch_yolo_total | 10.872 |
| branch_audio_total | 50.257 |
