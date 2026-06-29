# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 08:21:10 UTC | iJJprA7tnLA_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 175.810 | 0.676 | 52.807 | 18.267 | 18.034 | 25.018 | 3.928 |

## 2026-06-26 08:21:10 UTC | iJJprA7tnLA_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/iJJprA7tnLA_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `175.810` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.676 |
| save_clips | - |
| sample_frames | 0.967 |
| caption_frames | 43.450 |
| sample_fps | 2.134 |
| detect_object_yolo | 9.138 |
| audio_scan | 8.609 |
| asr_timings | 11.600 |
| ast_timings | 32.589 |
| describe_scenes | 18.267 |
| summarize_scenes | 18.034 |
| synthesize_synopsis | 25.018 |
| make_embedding | 3.928 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.423 |
| branch_yolo_total | 11.278 |
| branch_audio_total | 52.807 |
