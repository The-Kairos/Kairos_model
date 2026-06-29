# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 19:11:28 UTC | AV9a_rRCOaU_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 175.588 | 0.915 | 60.976 | 18.768 | 10.016 | 14.396 | 4.499 |

## 2026-06-24 19:11:28 UTC | AV9a_rRCOaU_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/AV9a_rRCOaU_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `175.588` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.915 |
| save_clips | - |
| sample_frames | 1.468 |
| caption_frames | 50.482 |
| sample_fps | 2.515 |
| detect_object_yolo | 10.162 |
| audio_scan | 13.830 |
| asr_timings | 9.546 |
| ast_timings | 37.592 |
| describe_scenes | 18.768 |
| summarize_scenes | 10.016 |
| synthesize_synopsis | 14.396 |
| make_embedding | 4.499 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.956 |
| branch_yolo_total | 12.682 |
| branch_audio_total | 60.976 |
