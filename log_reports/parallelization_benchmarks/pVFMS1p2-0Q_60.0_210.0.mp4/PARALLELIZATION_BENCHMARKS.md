# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 08:11:08 UTC | pVFMS1p2-0Q_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 107.227 | 0.756 | 40.662 | 8.647 | 11.060 | 6.408 | 2.529 |

## 2026-06-28 08:11:08 UTC | pVFMS1p2-0Q_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/pVFMS1p2-0Q_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `107.227` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.756 |
| save_clips | - |
| sample_frames | 0.618 |
| caption_frames | 26.589 |
| sample_fps | 1.993 |
| detect_object_yolo | 6.592 |
| audio_scan | 12.758 |
| asr_timings | 9.157 |
| ast_timings | 18.737 |
| describe_scenes | 8.647 |
| summarize_scenes | 11.060 |
| synthesize_synopsis | 6.408 |
| make_embedding | 2.529 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 27.213 |
| branch_yolo_total | 8.591 |
| branch_audio_total | 40.662 |
