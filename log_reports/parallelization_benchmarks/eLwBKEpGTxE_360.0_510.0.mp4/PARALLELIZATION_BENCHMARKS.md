# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 03:42:32 UTC | eLwBKEpGTxE_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 151.906 | 0.793 | 58.120 | 14.371 | 8.292 | 8.699 | 3.889 |

## 2026-06-26 03:42:32 UTC | eLwBKEpGTxE_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/eLwBKEpGTxE_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `151.906` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.793 |
| save_clips | - |
| sample_frames | 1.103 |
| caption_frames | 43.077 |
| sample_fps | 2.323 |
| detect_object_yolo | 9.800 |
| audio_scan | 14.176 |
| asr_timings | 11.599 |
| ast_timings | 32.337 |
| describe_scenes | 14.371 |
| summarize_scenes | 8.292 |
| synthesize_synopsis | 8.699 |
| make_embedding | 3.889 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.186 |
| branch_yolo_total | 12.130 |
| branch_audio_total | 58.120 |
