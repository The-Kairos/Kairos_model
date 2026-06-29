# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 06:36:08 UTC | zvEaxc9WD14_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 182.617 | 0.783 | 69.283 | 15.143 | 15.054 | 4.806 | 5.381 |

## 2026-06-27 06:36:08 UTC | zvEaxc9WD14_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/zvEaxc9WD14_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `182.617` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.783 |
| save_clips | - |
| sample_frames | 1.408 |
| caption_frames | 55.738 |
| sample_fps | 2.495 |
| detect_object_yolo | 11.107 |
| audio_scan | 13.978 |
| asr_timings | 11.786 |
| ast_timings | 43.510 |
| describe_scenes | 15.143 |
| summarize_scenes | 15.054 |
| synthesize_synopsis | 4.806 |
| make_embedding | 5.381 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 57.152 |
| branch_yolo_total | 13.609 |
| branch_audio_total | 69.283 |
