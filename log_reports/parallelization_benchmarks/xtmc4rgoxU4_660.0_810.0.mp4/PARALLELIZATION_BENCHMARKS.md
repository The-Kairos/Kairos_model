# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 04:25:18 UTC | xtmc4rgoxU4_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 153.040 | 0.772 | 57.607 | 10.708 | 12.806 | 6.644 | 4.090 |

## 2026-06-27 04:25:18 UTC | xtmc4rgoxU4_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/xtmc4rgoxU4_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `153.040` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.772 |
| save_clips | - |
| sample_frames | 1.198 |
| caption_frames | 46.089 |
| sample_fps | 2.325 |
| detect_object_yolo | 9.397 |
| audio_scan | 10.917 |
| asr_timings | 11.353 |
| ast_timings | 35.328 |
| describe_scenes | 10.708 |
| summarize_scenes | 12.806 |
| synthesize_synopsis | 6.644 |
| make_embedding | 4.090 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.292 |
| branch_yolo_total | 11.728 |
| branch_audio_total | 57.607 |
