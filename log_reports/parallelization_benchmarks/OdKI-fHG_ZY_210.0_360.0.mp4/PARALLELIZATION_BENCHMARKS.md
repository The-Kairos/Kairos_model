# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 12:03:12 UTC | OdKI-fHG_ZY_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 224.275 | 0.675 | 65.260 | 31.584 | 25.222 | 24.539 | 5.327 |

## 2026-06-25 12:03:12 UTC | OdKI-fHG_ZY_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/OdKI-fHG_ZY_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `224.275` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.675 |
| save_clips | - |
| sample_frames | 1.346 |
| caption_frames | 55.165 |
| sample_fps | 2.391 |
| detect_object_yolo | 11.327 |
| audio_scan | 9.977 |
| asr_timings | 9.780 |
| ast_timings | 45.494 |
| describe_scenes | 31.584 |
| summarize_scenes | 25.222 |
| synthesize_synopsis | 24.539 |
| make_embedding | 5.327 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 56.516 |
| branch_yolo_total | 13.724 |
| branch_audio_total | 65.260 |
