# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 16:10:56 UTC | mnSYg_uCdtE_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 149.074 | 0.761 | 53.280 | 17.327 | 9.807 | 10.736 | 3.966 |

## 2026-06-27 16:10:56 UTC | mnSYg_uCdtE_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/mnSYg_uCdtE_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `149.074` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.761 |
| save_clips | - |
| sample_frames | 1.092 |
| caption_frames | 39.065 |
| sample_fps | 2.269 |
| detect_object_yolo | 9.399 |
| audio_scan | 12.672 |
| asr_timings | 8.141 |
| ast_timings | 32.459 |
| describe_scenes | 17.327 |
| summarize_scenes | 9.807 |
| synthesize_synopsis | 10.736 |
| make_embedding | 3.966 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 40.163 |
| branch_yolo_total | 11.674 |
| branch_audio_total | 53.280 |
