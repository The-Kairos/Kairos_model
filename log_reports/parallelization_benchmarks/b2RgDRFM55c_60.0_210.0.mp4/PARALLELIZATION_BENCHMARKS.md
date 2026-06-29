# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 00:48:26 UTC | b2RgDRFM55c_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 209.131 | 0.671 | 89.296 | 18.720 | 14.241 | 7.386 | 5.351 |

## 2026-06-26 00:48:26 UTC | b2RgDRFM55c_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/b2RgDRFM55c_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `209.131` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.671 |
| save_clips | - |
| sample_frames | 1.476 |
| caption_frames | 57.067 |
| sample_fps | 2.375 |
| detect_object_yolo | 11.140 |
| audio_scan | 12.785 |
| asr_timings | 31.911 |
| ast_timings | 44.591 |
| describe_scenes | 18.720 |
| summarize_scenes | 14.241 |
| synthesize_synopsis | 7.386 |
| make_embedding | 5.351 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 58.548 |
| branch_yolo_total | 13.520 |
| branch_audio_total | 89.296 |
