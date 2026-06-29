# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 18:31:44 UTC | 9fJEFi3ccwI_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 221.442 | 0.915 | 77.330 | 20.552 | 16.287 | 13.928 | 6.343 |

## 2026-06-24 18:31:44 UTC | 9fJEFi3ccwI_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/9fJEFi3ccwI_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `221.442` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.915 |
| save_clips | - |
| sample_frames | 2.287 |
| caption_frames | 67.471 |
| sample_fps | 3.051 |
| detect_object_yolo | 11.874 |
| audio_scan | 15.924 |
| asr_timings | 9.177 |
| ast_timings | 52.221 |
| describe_scenes | 20.552 |
| summarize_scenes | 16.287 |
| synthesize_synopsis | 13.928 |
| make_embedding | 6.343 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 69.764 |
| branch_yolo_total | 14.931 |
| branch_audio_total | 77.330 |
