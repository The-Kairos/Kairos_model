# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 22:38:17 UTC | sqFIsmskWaw_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 157.029 | 0.793 | 56.596 | 14.781 | 12.807 | 6.281 | 4.076 |

## 2026-06-26 22:38:17 UTC | sqFIsmskWaw_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/sqFIsmskWaw_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `157.029` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.793 |
| save_clips | - |
| sample_frames | 1.574 |
| caption_frames | 46.276 |
| sample_fps | 2.380 |
| detect_object_yolo | 10.044 |
| audio_scan | 11.666 |
| asr_timings | 9.795 |
| ast_timings | 35.126 |
| describe_scenes | 14.781 |
| summarize_scenes | 12.807 |
| synthesize_synopsis | 6.281 |
| make_embedding | 4.076 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.856 |
| branch_yolo_total | 12.430 |
| branch_audio_total | 56.596 |
