# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 08:22:47 UTC | q2tnZxvcE7A_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 210.626 | 0.691 | 97.730 | 13.531 | 10.798 | 7.511 | 5.440 |

## 2026-06-28 08:22:47 UTC | q2tnZxvcE7A_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/q2tnZxvcE7A_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `210.626` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.691 |
| save_clips | - |
| sample_frames | 2.320 |
| caption_frames | 57.410 |
| sample_fps | 2.680 |
| detect_object_yolo | 11.110 |
| audio_scan | 12.789 |
| asr_timings | 40.859 |
| ast_timings | 44.073 |
| describe_scenes | 13.531 |
| summarize_scenes | 10.798 |
| synthesize_synopsis | 7.511 |
| make_embedding | 5.440 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 59.736 |
| branch_yolo_total | 13.796 |
| branch_audio_total | 97.730 |
