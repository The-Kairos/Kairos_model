# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 09:12:19 UTC | qqPwV3d4BZE_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 140.274 | 0.782 | 55.758 | 9.549 | 8.050 | 8.472 | 3.568 |

## 2026-06-28 09:12:19 UTC | qqPwV3d4BZE_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/qqPwV3d4BZE_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `140.274` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.782 |
| save_clips | - |
| sample_frames | 1.063 |
| caption_frames | 40.356 |
| sample_fps | 2.286 |
| detect_object_yolo | 9.006 |
| audio_scan | 14.876 |
| asr_timings | 11.380 |
| ast_timings | 29.493 |
| describe_scenes | 9.549 |
| summarize_scenes | 8.050 |
| synthesize_synopsis | 8.472 |
| make_embedding | 3.568 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.425 |
| branch_yolo_total | 11.297 |
| branch_audio_total | 55.758 |
