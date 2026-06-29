# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 22:45:22 UTC | DRTht2xuV-k_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 199.922 | 0.776 | 99.734 | 13.931 | 6.583 | 9.845 | 4.386 |

## 2026-06-24 22:45:22 UTC | DRTht2xuV-k_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/DRTht2xuV-k_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `199.922` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.776 |
| save_clips | - |
| sample_frames | 1.307 |
| caption_frames | 49.201 |
| sample_fps | 2.404 |
| detect_object_yolo | 10.326 |
| audio_scan | 15.085 |
| asr_timings | 46.778 |
| ast_timings | 37.863 |
| describe_scenes | 13.931 |
| summarize_scenes | 6.583 |
| synthesize_synopsis | 9.845 |
| make_embedding | 4.386 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 50.515 |
| branch_yolo_total | 12.736 |
| branch_audio_total | 99.734 |
