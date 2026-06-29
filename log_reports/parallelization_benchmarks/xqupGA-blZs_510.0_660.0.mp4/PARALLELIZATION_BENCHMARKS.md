# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 04:10:47 UTC | xqupGA-blZs_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 198.323 | 0.769 | 82.388 | 16.190 | 13.520 | 10.156 | 5.041 |

## 2026-06-27 04:10:47 UTC | xqupGA-blZs_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/xqupGA-blZs_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `198.323` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.769 |
| save_clips | - |
| sample_frames | 1.434 |
| caption_frames | 54.193 |
| sample_fps | 2.493 |
| detect_object_yolo | 10.695 |
| audio_scan | 15.382 |
| asr_timings | 24.876 |
| ast_timings | 42.122 |
| describe_scenes | 16.190 |
| summarize_scenes | 13.520 |
| synthesize_synopsis | 10.156 |
| make_embedding | 5.041 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 55.633 |
| branch_yolo_total | 13.194 |
| branch_audio_total | 82.388 |
