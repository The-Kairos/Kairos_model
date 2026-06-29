# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 19:08:32 UTC | AV9a_rRCOaU_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 187.591 | 0.806 | 60.200 | 21.919 | 12.125 | 14.986 | 5.037 |

## 2026-06-24 19:08:32 UTC | AV9a_rRCOaU_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/AV9a_rRCOaU_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `187.591` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.806 |
| save_clips | - |
| sample_frames | 1.872 |
| caption_frames | 55.466 |
| sample_fps | 2.667 |
| detect_object_yolo | 11.092 |
| audio_scan | 8.680 |
| asr_timings | 10.184 |
| ast_timings | 41.328 |
| describe_scenes | 21.919 |
| summarize_scenes | 12.125 |
| synthesize_synopsis | 14.986 |
| make_embedding | 5.037 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 57.343 |
| branch_yolo_total | 13.765 |
| branch_audio_total | 60.200 |
