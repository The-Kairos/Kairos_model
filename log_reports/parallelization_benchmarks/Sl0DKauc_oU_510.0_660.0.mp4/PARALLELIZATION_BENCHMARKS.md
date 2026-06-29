# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 17:28:50 UTC | Sl0DKauc_oU_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 230.093 | 0.818 | 73.655 | 20.489 | 21.888 | 15.706 | 7.032 |

## 2026-06-25 17:28:50 UTC | Sl0DKauc_oU_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Sl0DKauc_oU_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `230.093` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.818 |
| save_clips | - |
| sample_frames | 1.803 |
| caption_frames | 71.846 |
| sample_fps | 2.683 |
| detect_object_yolo | 13.039 |
| audio_scan | 10.595 |
| asr_timings | 8.572 |
| ast_timings | 54.174 |
| describe_scenes | 20.489 |
| summarize_scenes | 21.888 |
| synthesize_synopsis | 15.706 |
| make_embedding | 7.032 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 73.655 |
| branch_yolo_total | 15.728 |
| branch_audio_total | 73.350 |
