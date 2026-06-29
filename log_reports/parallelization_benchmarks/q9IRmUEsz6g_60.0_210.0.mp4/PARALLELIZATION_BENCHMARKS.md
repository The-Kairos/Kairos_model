# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 08:45:48 UTC | q9IRmUEsz6g_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 195.281 | 0.705 | 71.993 | 15.219 | 13.626 | 8.510 | 5.702 |

## 2026-06-28 08:45:48 UTC | q9IRmUEsz6g_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/q9IRmUEsz6g_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `195.281` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.705 |
| save_clips | - |
| sample_frames | 1.717 |
| caption_frames | 62.071 |
| sample_fps | 2.494 |
| detect_object_yolo | 11.793 |
| audio_scan | 14.983 |
| asr_timings | 10.584 |
| ast_timings | 46.417 |
| describe_scenes | 15.219 |
| summarize_scenes | 13.626 |
| synthesize_synopsis | 8.510 |
| make_embedding | 5.702 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 63.794 |
| branch_yolo_total | 14.293 |
| branch_audio_total | 71.993 |
