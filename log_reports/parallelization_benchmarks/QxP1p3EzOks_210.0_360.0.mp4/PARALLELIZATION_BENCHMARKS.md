# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 16:03:45 UTC | QxP1p3EzOks_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 170.425 | 0.696 | 59.255 | 22.931 | 11.780 | 12.023 | 4.196 |

## 2026-06-25 16:03:45 UTC | QxP1p3EzOks_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/QxP1p3EzOks_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `170.425` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.696 |
| save_clips | - |
| sample_frames | 1.092 |
| caption_frames | 45.141 |
| sample_fps | 2.166 |
| detect_object_yolo | 9.712 |
| audio_scan | 14.517 |
| asr_timings | 9.386 |
| ast_timings | 35.343 |
| describe_scenes | 22.931 |
| summarize_scenes | 11.780 |
| synthesize_synopsis | 12.023 |
| make_embedding | 4.196 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.239 |
| branch_yolo_total | 11.884 |
| branch_audio_total | 59.255 |
