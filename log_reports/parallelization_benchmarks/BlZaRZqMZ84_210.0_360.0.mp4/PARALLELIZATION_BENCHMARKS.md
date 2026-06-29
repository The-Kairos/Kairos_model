# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 19:44:18 UTC | BlZaRZqMZ84_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 95.398 | 0.672 | 36.824 | 6.818 | 5.048 | 14.598 | 2.209 |

## 2026-06-24 19:44:18 UTC | BlZaRZqMZ84_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/BlZaRZqMZ84_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `95.398` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.672 |
| save_clips | - |
| sample_frames | 0.417 |
| caption_frames | 19.405 |
| sample_fps | 1.762 |
| detect_object_yolo | 6.229 |
| audio_scan | 14.938 |
| asr_timings | 9.111 |
| ast_timings | 12.766 |
| describe_scenes | 6.818 |
| summarize_scenes | 5.048 |
| synthesize_synopsis | 14.598 |
| make_embedding | 2.209 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 19.827 |
| branch_yolo_total | 7.998 |
| branch_audio_total | 36.824 |
