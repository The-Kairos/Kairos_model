# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 03:18:55 UTC | di-kKBvebi8_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 163.051 | 0.786 | 56.689 | 13.269 | 13.240 | 17.132 | 3.863 |

## 2026-06-26 03:18:55 UTC | di-kKBvebi8_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/di-kKBvebi8_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `163.051` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.786 |
| save_clips | - |
| sample_frames | 1.253 |
| caption_frames | 43.515 |
| sample_fps | 2.330 |
| detect_object_yolo | 9.575 |
| audio_scan | 14.137 |
| asr_timings | 10.941 |
| ast_timings | 31.603 |
| describe_scenes | 13.269 |
| summarize_scenes | 13.240 |
| synthesize_synopsis | 17.132 |
| make_embedding | 3.863 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.775 |
| branch_yolo_total | 11.911 |
| branch_audio_total | 56.689 |
