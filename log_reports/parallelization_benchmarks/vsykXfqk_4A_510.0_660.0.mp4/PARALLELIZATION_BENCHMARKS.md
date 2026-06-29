# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 02:40:56 UTC | vsykXfqk_4A_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 173.249 | 0.816 | 61.236 | 14.363 | 12.228 | 7.534 | 5.421 |

## 2026-06-27 02:40:56 UTC | vsykXfqk_4A_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/vsykXfqk_4A_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `173.249` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.816 |
| save_clips | - |
| sample_frames | 1.468 |
| caption_frames | 55.467 |
| sample_fps | 2.552 |
| detect_object_yolo | 10.750 |
| audio_scan | 9.761 |
| asr_timings | 7.058 |
| ast_timings | 44.409 |
| describe_scenes | 14.363 |
| summarize_scenes | 12.228 |
| synthesize_synopsis | 7.534 |
| make_embedding | 5.421 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 56.941 |
| branch_yolo_total | 13.308 |
| branch_audio_total | 61.236 |
