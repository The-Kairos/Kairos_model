# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 09:13:58 UTC | qqPwV3d4BZE_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 98.004 | 0.786 | 41.336 | 6.200 | 5.589 | 8.948 | 2.302 |

## 2026-06-28 09:13:58 UTC | qqPwV3d4BZE_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/qqPwV3d4BZE_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `98.004` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.786 |
| save_clips | - |
| sample_frames | 0.636 |
| caption_frames | 21.863 |
| sample_fps | 2.019 |
| detect_object_yolo | 6.930 |
| audio_scan | 13.834 |
| asr_timings | 11.930 |
| ast_timings | 15.563 |
| describe_scenes | 6.200 |
| summarize_scenes | 5.589 |
| synthesize_synopsis | 8.948 |
| make_embedding | 2.302 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 22.505 |
| branch_yolo_total | 8.955 |
| branch_audio_total | 41.336 |
