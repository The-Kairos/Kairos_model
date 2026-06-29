# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 22:10:25 UTC | Z2eyocRl2mo_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 111.841 | 0.643 | 40.559 | 9.134 | 7.895 | 9.080 | 2.785 |

## 2026-06-25 22:10:25 UTC | Z2eyocRl2mo_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Z2eyocRl2mo_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `111.841` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.643 |
| save_clips | - |
| sample_frames | 0.756 |
| caption_frames | 29.677 |
| sample_fps | 1.987 |
| detect_object_yolo | 7.944 |
| audio_scan | 6.455 |
| asr_timings | 12.294 |
| ast_timings | 21.801 |
| describe_scenes | 9.134 |
| summarize_scenes | 7.895 |
| synthesize_synopsis | 9.080 |
| make_embedding | 2.785 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.438 |
| branch_yolo_total | 9.937 |
| branch_audio_total | 40.559 |
