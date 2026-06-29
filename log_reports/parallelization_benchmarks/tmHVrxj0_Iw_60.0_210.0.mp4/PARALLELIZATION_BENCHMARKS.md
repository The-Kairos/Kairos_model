# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 23:52:30 UTC | tmHVrxj0_Iw_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 180.848 | 0.776 | 61.317 | 15.662 | 21.020 | 6.648 | 5.085 |

## 2026-06-26 23:52:30 UTC | tmHVrxj0_Iw_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/tmHVrxj0_Iw_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `180.848` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.776 |
| save_clips | - |
| sample_frames | 1.463 |
| caption_frames | 54.219 |
| sample_fps | 2.469 |
| detect_object_yolo | 10.761 |
| audio_scan | 15.032 |
| asr_timings | 7.726 |
| ast_timings | 38.550 |
| describe_scenes | 15.662 |
| summarize_scenes | 21.020 |
| synthesize_synopsis | 6.648 |
| make_embedding | 5.085 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 55.689 |
| branch_yolo_total | 13.236 |
| branch_audio_total | 61.317 |
