# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 10:08:00 UTC | j7g-XYX5FRc_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 182.601 | 0.799 | 55.992 | 15.440 | 16.840 | 36.436 | 3.594 |

## 2026-06-26 10:08:00 UTC | j7g-XYX5FRc_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/j7g-XYX5FRc_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `182.601` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.799 |
| save_clips | - |
| sample_frames | 0.826 |
| caption_frames | 39.533 |
| sample_fps | 2.191 |
| detect_object_yolo | 9.529 |
| audio_scan | 16.201 |
| asr_timings | 10.305 |
| ast_timings | 29.476 |
| describe_scenes | 15.440 |
| summarize_scenes | 16.840 |
| synthesize_synopsis | 36.436 |
| make_embedding | 3.594 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 40.365 |
| branch_yolo_total | 11.726 |
| branch_audio_total | 55.992 |
