# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 18:05:38 UTC | m8802lFkHOs_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 260.738 | 0.877 | 82.308 | 36.947 | 16.137 | 19.413 | 6.720 |

## 2026-06-26 18:05:38 UTC | m8802lFkHOs_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/m8802lFkHOs_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `260.738` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.877 |
| save_clips | - |
| sample_frames | 2.010 |
| caption_frames | 78.092 |
| sample_fps | 2.919 |
| detect_object_yolo | 13.841 |
| audio_scan | 13.963 |
| asr_timings | 10.744 |
| ast_timings | 57.592 |
| describe_scenes | 36.947 |
| summarize_scenes | 16.137 |
| synthesize_synopsis | 19.413 |
| make_embedding | 6.720 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 80.109 |
| branch_yolo_total | 16.766 |
| branch_audio_total | 82.308 |
