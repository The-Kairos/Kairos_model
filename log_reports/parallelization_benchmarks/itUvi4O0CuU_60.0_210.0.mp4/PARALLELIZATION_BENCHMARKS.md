# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 09:41:11 UTC | itUvi4O0CuU_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 230.564 | 0.771 | 92.627 | 25.315 | 22.976 | 17.230 | 4.543 |

## 2026-06-26 09:41:11 UTC | itUvi4O0CuU_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/itUvi4O0CuU_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `230.564` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.771 |
| save_clips | - |
| sample_frames | 1.421 |
| caption_frames | 51.299 |
| sample_fps | 2.370 |
| detect_object_yolo | 10.552 |
| audio_scan | 14.152 |
| asr_timings | 40.319 |
| ast_timings | 38.146 |
| describe_scenes | 25.315 |
| summarize_scenes | 22.976 |
| synthesize_synopsis | 17.230 |
| make_embedding | 4.543 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 52.726 |
| branch_yolo_total | 12.928 |
| branch_audio_total | 92.627 |
