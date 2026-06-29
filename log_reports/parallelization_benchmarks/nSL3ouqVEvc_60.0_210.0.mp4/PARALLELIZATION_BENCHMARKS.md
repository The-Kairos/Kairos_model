# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 16:35:06 UTC | nSL3ouqVEvc_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 162.663 | 0.649 | 59.031 | 11.605 | 17.261 | 9.563 | 4.350 |

## 2026-06-27 16:35:06 UTC | nSL3ouqVEvc_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/nSL3ouqVEvc_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `162.663` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.649 |
| save_clips | - |
| sample_frames | 1.135 |
| caption_frames | 45.697 |
| sample_fps | 2.237 |
| detect_object_yolo | 9.755 |
| audio_scan | 14.763 |
| asr_timings | 9.856 |
| ast_timings | 34.404 |
| describe_scenes | 11.605 |
| summarize_scenes | 17.261 |
| synthesize_synopsis | 9.563 |
| make_embedding | 4.350 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.838 |
| branch_yolo_total | 11.998 |
| branch_audio_total | 59.031 |
