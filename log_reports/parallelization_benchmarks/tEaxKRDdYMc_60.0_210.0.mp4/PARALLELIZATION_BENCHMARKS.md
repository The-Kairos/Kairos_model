# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 23:00:40 UTC | tEaxKRDdYMc_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 97.390 | 0.643 | 38.732 | 8.430 | 5.098 | 8.035 | 2.313 |

## 2026-06-26 23:00:40 UTC | tEaxKRDdYMc_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/tEaxKRDdYMc_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `97.390` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.643 |
| save_clips | - |
| sample_frames | 0.572 |
| caption_frames | 23.026 |
| sample_fps | 1.900 |
| detect_object_yolo | 7.195 |
| audio_scan | 12.890 |
| asr_timings | 8.756 |
| ast_timings | 17.077 |
| describe_scenes | 8.430 |
| summarize_scenes | 5.098 |
| synthesize_synopsis | 8.035 |
| make_embedding | 2.313 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 23.604 |
| branch_yolo_total | 9.101 |
| branch_audio_total | 38.732 |
