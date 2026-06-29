# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 08:14:54 UTC | iCbjk3uRtqA_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 147.807 | 0.786 | 47.781 | 13.554 | 21.305 | 20.549 | 2.809 |

## 2026-06-26 08:14:54 UTC | iCbjk3uRtqA_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/iCbjk3uRtqA_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `147.807` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.786 |
| save_clips | - |
| sample_frames | 0.724 |
| caption_frames | 28.879 |
| sample_fps | 2.050 |
| detect_object_yolo | 7.954 |
| audio_scan | 16.180 |
| asr_timings | 10.563 |
| ast_timings | 21.030 |
| describe_scenes | 13.554 |
| summarize_scenes | 21.305 |
| synthesize_synopsis | 20.549 |
| make_embedding | 2.809 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 29.610 |
| branch_yolo_total | 10.010 |
| branch_audio_total | 47.781 |
