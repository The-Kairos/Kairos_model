# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 21:35:53 UTC | XJlrte91c4A_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 139.936 | 0.786 | 48.744 | 13.740 | 12.771 | 8.399 | 3.512 |

## 2026-06-25 21:35:53 UTC | XJlrte91c4A_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/XJlrte91c4A_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `139.936` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.786 |
| save_clips | - |
| sample_frames | 0.877 |
| caption_frames | 38.945 |
| sample_fps | 2.229 |
| detect_object_yolo | 8.531 |
| audio_scan | 10.800 |
| asr_timings | 7.692 |
| ast_timings | 30.244 |
| describe_scenes | 13.740 |
| summarize_scenes | 12.771 |
| synthesize_synopsis | 8.399 |
| make_embedding | 3.512 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.827 |
| branch_yolo_total | 10.766 |
| branch_audio_total | 48.744 |
