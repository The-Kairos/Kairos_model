# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 00:10:10 UTC | a13cXTeCius_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 166.922 | 0.798 | 60.228 | 13.386 | 18.287 | 13.467 | 3.868 |

## 2026-06-26 00:10:10 UTC | a13cXTeCius_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/a13cXTeCius_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `166.922` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.798 |
| save_clips | - |
| sample_frames | 1.085 |
| caption_frames | 42.731 |
| sample_fps | 2.272 |
| detect_object_yolo | 9.394 |
| audio_scan | 14.896 |
| asr_timings | 11.979 |
| ast_timings | 33.344 |
| describe_scenes | 13.386 |
| summarize_scenes | 18.287 |
| synthesize_synopsis | 13.467 |
| make_embedding | 3.868 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.823 |
| branch_yolo_total | 11.672 |
| branch_audio_total | 60.228 |
