# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 14:45:37 UTC | l5cTU4dhUGY_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 121.882 | 0.661 | 38.258 | 4.709 | 29.181 | 15.926 | 2.090 |

## 2026-06-26 14:45:37 UTC | l5cTU4dhUGY_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/l5cTU4dhUGY_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `121.882` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.661 |
| save_clips | - |
| sample_frames | 0.569 |
| caption_frames | 20.183 |
| sample_fps | 1.832 |
| detect_object_yolo | 7.037 |
| audio_scan | 15.175 |
| asr_timings | 9.802 |
| ast_timings | 13.273 |
| describe_scenes | 4.709 |
| summarize_scenes | 29.181 |
| synthesize_synopsis | 15.926 |
| make_embedding | 2.090 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 20.758 |
| branch_yolo_total | 8.874 |
| branch_audio_total | 38.258 |
