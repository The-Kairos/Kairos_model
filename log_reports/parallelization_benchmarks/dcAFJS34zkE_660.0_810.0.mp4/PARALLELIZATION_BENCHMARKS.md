# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 03:13:48 UTC | dcAFJS34zkE_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 128.248 | 0.785 | 39.660 | 9.329 | 16.908 | 16.258 | 2.762 |

## 2026-06-26 03:13:48 UTC | dcAFJS34zkE_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/dcAFJS34zkE_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `128.248` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.785 |
| save_clips | - |
| sample_frames | 0.746 |
| caption_frames | 30.890 |
| sample_fps | 2.099 |
| detect_object_yolo | 7.409 |
| audio_scan | 5.455 |
| asr_timings | 12.305 |
| ast_timings | 21.890 |
| describe_scenes | 9.329 |
| summarize_scenes | 16.908 |
| synthesize_synopsis | 16.258 |
| make_embedding | 2.762 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 31.642 |
| branch_yolo_total | 9.514 |
| branch_audio_total | 39.660 |
