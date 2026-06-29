# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 03:48:04 UTC | eLwBKEpGTxE_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 165.252 | 0.839 | 58.341 | 13.232 | 14.235 | 15.300 | 3.968 |

## 2026-06-26 03:48:04 UTC | eLwBKEpGTxE_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/eLwBKEpGTxE_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `165.252` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.839 |
| save_clips | - |
| sample_frames | 1.307 |
| caption_frames | 44.181 |
| sample_fps | 2.437 |
| detect_object_yolo | 9.927 |
| audio_scan | 13.167 |
| asr_timings | 12.200 |
| ast_timings | 32.967 |
| describe_scenes | 13.232 |
| summarize_scenes | 14.235 |
| synthesize_synopsis | 15.300 |
| make_embedding | 3.968 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.494 |
| branch_yolo_total | 12.370 |
| branch_audio_total | 58.341 |
