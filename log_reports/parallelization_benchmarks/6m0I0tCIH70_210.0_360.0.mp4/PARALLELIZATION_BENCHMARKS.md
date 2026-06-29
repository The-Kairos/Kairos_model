# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 12:49:01 UTC | 6m0I0tCIH70_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 135.592 | 0.791 | 43.030 | 10.431 | 9.870 | 17.174 | 3.606 |

## 2026-06-24 12:49:01 UTC | 6m0I0tCIH70_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/6m0I0tCIH70_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `135.592` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.791 |
| save_clips | - |
| sample_frames | 1.131 |
| caption_frames | 37.728 |
| sample_fps | 2.273 |
| detect_object_yolo | 8.174 |
| audio_scan | 7.482 |
| asr_timings | 11.506 |
| ast_timings | 24.033 |
| describe_scenes | 10.431 |
| summarize_scenes | 9.870 |
| synthesize_synopsis | 17.174 |
| make_embedding | 3.606 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.865 |
| branch_yolo_total | 10.453 |
| branch_audio_total | 43.030 |
