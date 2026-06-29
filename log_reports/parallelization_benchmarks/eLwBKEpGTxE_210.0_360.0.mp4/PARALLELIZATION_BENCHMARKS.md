# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 03:39:59 UTC | eLwBKEpGTxE_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 158.502 | 0.818 | 60.034 | 10.784 | 14.201 | 7.895 | 4.160 |

## 2026-06-26 03:39:59 UTC | eLwBKEpGTxE_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/eLwBKEpGTxE_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `158.502` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.818 |
| save_clips | - |
| sample_frames | 1.617 |
| caption_frames | 45.100 |
| sample_fps | 2.470 |
| detect_object_yolo | 9.980 |
| audio_scan | 14.186 |
| asr_timings | 10.413 |
| ast_timings | 35.427 |
| describe_scenes | 10.784 |
| summarize_scenes | 14.201 |
| synthesize_synopsis | 7.895 |
| make_embedding | 4.160 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.723 |
| branch_yolo_total | 12.455 |
| branch_audio_total | 60.034 |
