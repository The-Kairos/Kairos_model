# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 02:31:32 UTC | viPIq7-BdpU_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 149.397 | 0.657 | 53.305 | 10.481 | 18.594 | 9.976 | 4.845 |

## 2026-06-27 02:31:32 UTC | viPIq7-BdpU_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/viPIq7-BdpU_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `149.397` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.657 |
| save_clips | - |
| sample_frames | 0.856 |
| caption_frames | 38.495 |
| sample_fps | 2.038 |
| detect_object_yolo | 8.733 |
| audio_scan | 16.079 |
| asr_timings | 10.037 |
| ast_timings | 27.180 |
| describe_scenes | 10.481 |
| summarize_scenes | 18.594 |
| synthesize_synopsis | 9.976 |
| make_embedding | 4.845 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.358 |
| branch_yolo_total | 10.777 |
| branch_audio_total | 53.305 |
