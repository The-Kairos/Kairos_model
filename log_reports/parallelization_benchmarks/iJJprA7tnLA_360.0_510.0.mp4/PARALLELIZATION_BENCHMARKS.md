# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 08:23:24 UTC | iJJprA7tnLA_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 132.907 | 0.682 | 42.590 | 13.992 | 11.095 | 23.133 | 2.651 |

## 2026-06-26 08:23:24 UTC | iJJprA7tnLA_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/iJJprA7tnLA_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `132.907` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.682 |
| save_clips | - |
| sample_frames | 0.663 |
| caption_frames | 27.272 |
| sample_fps | 1.928 |
| detect_object_yolo | 7.504 |
| audio_scan | 14.031 |
| asr_timings | 9.676 |
| ast_timings | 18.874 |
| describe_scenes | 13.992 |
| summarize_scenes | 11.095 |
| synthesize_synopsis | 23.133 |
| make_embedding | 2.651 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 27.941 |
| branch_yolo_total | 9.437 |
| branch_audio_total | 42.590 |
