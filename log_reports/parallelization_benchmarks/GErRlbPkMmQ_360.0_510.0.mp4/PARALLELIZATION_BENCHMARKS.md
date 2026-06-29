# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 01:29:43 UTC | GErRlbPkMmQ_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 130.327 | 0.796 | 48.320 | 11.573 | 9.734 | 13.735 | 3.029 |

## 2026-06-25 01:29:43 UTC | GErRlbPkMmQ_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/GErRlbPkMmQ_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `130.327` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.796 |
| save_clips | - |
| sample_frames | 0.908 |
| caption_frames | 30.835 |
| sample_fps | 2.149 |
| detect_object_yolo | 7.852 |
| audio_scan | 15.751 |
| asr_timings | 8.071 |
| ast_timings | 24.489 |
| describe_scenes | 11.573 |
| summarize_scenes | 9.734 |
| synthesize_synopsis | 13.735 |
| make_embedding | 3.029 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 31.749 |
| branch_yolo_total | 10.007 |
| branch_audio_total | 48.320 |
