# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 16:06:41 UTC | QxP1p3EzOks_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 174.504 | 0.715 | 62.803 | 16.654 | 13.989 | 20.014 | 3.887 |

## 2026-06-25 16:06:41 UTC | QxP1p3EzOks_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/QxP1p3EzOks_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `174.504` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.715 |
| save_clips | - |
| sample_frames | 1.108 |
| caption_frames | 42.185 |
| sample_fps | 2.167 |
| detect_object_yolo | 9.478 |
| audio_scan | 16.773 |
| asr_timings | 13.204 |
| ast_timings | 32.818 |
| describe_scenes | 16.654 |
| summarize_scenes | 13.989 |
| synthesize_synopsis | 20.014 |
| make_embedding | 3.887 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.298 |
| branch_yolo_total | 11.651 |
| branch_audio_total | 62.803 |
