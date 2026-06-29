# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 10:04:57 UTC | izWkhAfNNQg_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 197.330 | 0.813 | 62.973 | 19.370 | 19.943 | 26.399 | 4.152 |

## 2026-06-26 10:04:57 UTC | izWkhAfNNQg_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/izWkhAfNNQg_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `197.330` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.813 |
| save_clips | - |
| sample_frames | 1.644 |
| caption_frames | 48.215 |
| sample_fps | 2.447 |
| detect_object_yolo | 9.951 |
| audio_scan | 16.105 |
| asr_timings | 12.297 |
| ast_timings | 34.563 |
| describe_scenes | 19.370 |
| summarize_scenes | 19.943 |
| synthesize_synopsis | 26.399 |
| make_embedding | 4.152 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.865 |
| branch_yolo_total | 12.404 |
| branch_audio_total | 62.973 |
