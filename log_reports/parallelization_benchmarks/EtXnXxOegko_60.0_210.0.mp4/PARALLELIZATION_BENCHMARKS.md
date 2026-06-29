# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 00:00:53 UTC | EtXnXxOegko_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 118.608 | 0.834 | 46.555 | 8.031 | 5.770 | 8.353 | 3.007 |

## 2026-06-25 00:00:53 UTC | EtXnXxOegko_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/EtXnXxOegko_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `118.608` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.834 |
| save_clips | - |
| sample_frames | 1.032 |
| caption_frames | 33.282 |
| sample_fps | 2.173 |
| detect_object_yolo | 8.135 |
| audio_scan | 11.860 |
| asr_timings | 10.471 |
| ast_timings | 24.215 |
| describe_scenes | 8.031 |
| summarize_scenes | 5.770 |
| synthesize_synopsis | 8.353 |
| make_embedding | 3.007 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 34.321 |
| branch_yolo_total | 10.314 |
| branch_audio_total | 46.555 |
