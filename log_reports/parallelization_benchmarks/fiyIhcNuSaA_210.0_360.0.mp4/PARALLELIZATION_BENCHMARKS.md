# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 04:31:26 UTC | fiyIhcNuSaA_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 90.035 | 1.604 | 37.201 | 4.621 | 6.922 | 9.035 | 2.094 |

## 2026-06-26 04:31:26 UTC | fiyIhcNuSaA_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/fiyIhcNuSaA_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `90.035` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.604 |
| save_clips | - |
| sample_frames | 0.507 |
| caption_frames | 17.968 |
| sample_fps | 1.973 |
| detect_object_yolo | 6.652 |
| audio_scan | 12.008 |
| asr_timings | 12.116 |
| ast_timings | 13.068 |
| describe_scenes | 4.621 |
| summarize_scenes | 6.922 |
| synthesize_synopsis | 9.035 |
| make_embedding | 2.094 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 18.480 |
| branch_yolo_total | 8.630 |
| branch_audio_total | 37.201 |
