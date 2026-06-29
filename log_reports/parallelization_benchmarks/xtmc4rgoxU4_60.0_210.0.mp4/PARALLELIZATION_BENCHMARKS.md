# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 04:22:44 UTC | xtmc4rgoxU4_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 182.014 | 0.783 | 75.403 | 15.046 | 12.525 | 8.586 | 4.474 |

## 2026-06-27 04:22:44 UTC | xtmc4rgoxU4_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/xtmc4rgoxU4_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `182.014` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.783 |
| save_clips | - |
| sample_frames | 1.285 |
| caption_frames | 50.154 |
| sample_fps | 2.387 |
| detect_object_yolo | 9.952 |
| audio_scan | 14.166 |
| asr_timings | 23.685 |
| ast_timings | 37.543 |
| describe_scenes | 15.046 |
| summarize_scenes | 12.525 |
| synthesize_synopsis | 8.586 |
| make_embedding | 4.474 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.445 |
| branch_yolo_total | 12.345 |
| branch_audio_total | 75.403 |
