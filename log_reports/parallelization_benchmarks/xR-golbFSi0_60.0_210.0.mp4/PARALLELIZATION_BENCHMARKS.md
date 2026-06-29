# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 03:54:50 UTC | xR-golbFSi0_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 75.175 | 0.813 | 28.059 | 4.821 | 7.649 | 9.867 | 1.558 |

## 2026-06-27 03:54:50 UTC | xR-golbFSi0_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/xR-golbFSi0_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `75.175` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.813 |
| save_clips | - |
| sample_frames | 0.246 |
| caption_frames | 12.757 |
| sample_fps | 1.870 |
| detect_object_yolo | 6.089 |
| audio_scan | 9.887 |
| asr_timings | 10.903 |
| ast_timings | 7.261 |
| describe_scenes | 4.821 |
| summarize_scenes | 7.649 |
| synthesize_synopsis | 9.867 |
| make_embedding | 1.558 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 13.009 |
| branch_yolo_total | 7.965 |
| branch_audio_total | 28.059 |
