# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 13:31:40 UTC | 0SkdELRCIpc_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 132.797 | 0.783 | 51.594 | 10.430 | 9.267 | 8.321 | 3.331 |

## 2026-06-27 13:31:40 UTC | 0SkdELRCIpc_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0SkdELRCIpc_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `132.797` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.783 |
| save_clips | - |
| sample_frames | 0.986 |
| caption_frames | 36.194 |
| sample_fps | 2.268 |
| detect_object_yolo | 8.209 |
| audio_scan | 14.928 |
| asr_timings | 9.582 |
| ast_timings | 27.076 |
| describe_scenes | 10.430 |
| summarize_scenes | 9.267 |
| synthesize_synopsis | 8.321 |
| make_embedding | 3.331 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.187 |
| branch_yolo_total | 10.483 |
| branch_audio_total | 51.594 |
