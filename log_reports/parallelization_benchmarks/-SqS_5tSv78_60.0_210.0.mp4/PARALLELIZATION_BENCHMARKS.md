# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 08:28:58 UTC | -SqS_5tSv78_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 200.661 | 0.789 | 61.593 | 34.669 | 15.661 | 21.362 | 4.282 |

## 2026-06-24 08:28:58 UTC | -SqS_5tSv78_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-SqS_5tSv78_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `200.661` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.789 |
| save_clips | - |
| sample_frames | 1.385 |
| caption_frames | 47.535 |
| sample_fps | 2.396 |
| detect_object_yolo | 9.636 |
| audio_scan | 14.793 |
| asr_timings | 10.851 |
| ast_timings | 35.941 |
| describe_scenes | 34.669 |
| summarize_scenes | 15.661 |
| synthesize_synopsis | 21.362 |
| make_embedding | 4.282 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 48.925 |
| branch_yolo_total | 12.038 |
| branch_audio_total | 61.593 |
