# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 00:02:45 UTC | EtXnXxOegko_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 110.554 | 0.793 | 39.792 | 6.436 | 6.462 | 15.736 | 2.509 |

## 2026-06-25 00:02:45 UTC | EtXnXxOegko_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/EtXnXxOegko_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `110.554` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.793 |
| save_clips | - |
| sample_frames | 0.814 |
| caption_frames | 27.199 |
| sample_fps | 2.087 |
| detect_object_yolo | 7.306 |
| audio_scan | 10.774 |
| asr_timings | 10.094 |
| ast_timings | 18.916 |
| describe_scenes | 6.436 |
| summarize_scenes | 6.462 |
| synthesize_synopsis | 15.736 |
| make_embedding | 2.509 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 28.019 |
| branch_yolo_total | 9.399 |
| branch_audio_total | 39.792 |
