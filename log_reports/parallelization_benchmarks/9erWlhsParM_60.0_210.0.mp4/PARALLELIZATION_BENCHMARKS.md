# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 18:18:06 UTC | 9erWlhsParM_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 99.215 | 0.623 | 37.593 | 5.228 | 7.418 | 22.217 | 1.798 |

## 2026-06-24 18:18:06 UTC | 9erWlhsParM_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/9erWlhsParM_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `99.215` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.623 |
| save_clips | - |
| sample_frames | 0.306 |
| caption_frames | 14.250 |
| sample_fps | 1.710 |
| detect_object_yolo | 6.710 |
| audio_scan | 14.976 |
| asr_timings | 12.287 |
| ast_timings | 10.322 |
| describe_scenes | 5.228 |
| summarize_scenes | 7.418 |
| synthesize_synopsis | 22.217 |
| make_embedding | 1.798 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 14.561 |
| branch_yolo_total | 8.425 |
| branch_audio_total | 37.593 |
