# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 19:59:35 UTC | BwG_4D8xuLk_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 150.240 | 0.792 | 49.445 | 12.452 | 22.120 | 12.706 | 3.315 |

## 2026-06-24 19:59:35 UTC | BwG_4D8xuLk_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/BwG_4D8xuLk_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `150.240` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.792 |
| save_clips | - |
| sample_frames | 1.005 |
| caption_frames | 36.376 |
| sample_fps | 2.190 |
| detect_object_yolo | 8.442 |
| audio_scan | 9.684 |
| asr_timings | 12.541 |
| ast_timings | 27.212 |
| describe_scenes | 12.452 |
| summarize_scenes | 22.120 |
| synthesize_synopsis | 12.706 |
| make_embedding | 3.315 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.387 |
| branch_yolo_total | 10.637 |
| branch_audio_total | 49.445 |
