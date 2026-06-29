# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 00:18:11 UTC | aIGR9knS1B0_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 123.452 | 0.698 | 39.023 | 13.302 | 10.473 | 9.717 | 3.555 |

## 2026-06-26 00:18:11 UTC | aIGR9knS1B0_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/aIGR9knS1B0_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `123.452` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.698 |
| save_clips | - |
| sample_frames | 1.175 |
| caption_frames | 37.842 |
| sample_fps | 2.191 |
| detect_object_yolo | 8.567 |
| audio_scan | 3.819 |
| asr_timings | 0.000 |
| ast_timings | 30.686 |
| describe_scenes | 13.302 |
| summarize_scenes | 10.473 |
| synthesize_synopsis | 9.717 |
| make_embedding | 3.555 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.023 |
| branch_yolo_total | 10.764 |
| branch_audio_total | 34.513 |
