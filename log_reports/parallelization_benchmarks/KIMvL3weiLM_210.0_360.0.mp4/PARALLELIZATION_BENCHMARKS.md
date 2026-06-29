# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 06:09:28 UTC | KIMvL3weiLM_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 153.197 | 0.698 | 51.902 | 15.508 | 11.494 | 18.483 | 3.360 |

## 2026-06-25 06:09:28 UTC | KIMvL3weiLM_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/KIMvL3weiLM_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `153.197` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.698 |
| save_clips | - |
| sample_frames | 0.953 |
| caption_frames | 38.502 |
| sample_fps | 2.070 |
| detect_object_yolo | 8.791 |
| audio_scan | 13.841 |
| asr_timings | 11.036 |
| ast_timings | 27.017 |
| describe_scenes | 15.508 |
| summarize_scenes | 11.494 |
| synthesize_synopsis | 18.483 |
| make_embedding | 3.360 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.461 |
| branch_yolo_total | 10.867 |
| branch_audio_total | 51.902 |
