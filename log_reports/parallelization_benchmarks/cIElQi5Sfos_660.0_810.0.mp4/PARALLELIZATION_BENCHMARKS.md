# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 02:20:33 UTC | cIElQi5Sfos_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 144.731 | 0.797 | 52.546 | 8.537 | 12.652 | 20.502 | 3.329 |

## 2026-06-26 02:20:33 UTC | cIElQi5Sfos_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/cIElQi5Sfos_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `144.731` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.797 |
| save_clips | - |
| sample_frames | 0.960 |
| caption_frames | 33.250 |
| sample_fps | 2.152 |
| detect_object_yolo | 8.607 |
| audio_scan | 12.993 |
| asr_timings | 12.176 |
| ast_timings | 27.369 |
| describe_scenes | 8.537 |
| summarize_scenes | 12.652 |
| synthesize_synopsis | 20.502 |
| make_embedding | 3.329 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 34.216 |
| branch_yolo_total | 10.765 |
| branch_audio_total | 52.546 |
