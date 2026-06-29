# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 02:37:44 UTC | ccMqbhacbpY_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 124.492 | 0.788 | 40.348 | 7.907 | 20.742 | 15.497 | 2.470 |

## 2026-06-26 02:37:44 UTC | ccMqbhacbpY_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ccMqbhacbpY_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `124.492` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.788 |
| save_clips | - |
| sample_frames | 0.709 |
| caption_frames | 25.758 |
| sample_fps | 2.028 |
| detect_object_yolo | 6.830 |
| audio_scan | 13.002 |
| asr_timings | 8.660 |
| ast_timings | 18.677 |
| describe_scenes | 7.907 |
| summarize_scenes | 20.742 |
| synthesize_synopsis | 15.497 |
| make_embedding | 2.470 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 26.474 |
| branch_yolo_total | 8.863 |
| branch_audio_total | 40.348 |
