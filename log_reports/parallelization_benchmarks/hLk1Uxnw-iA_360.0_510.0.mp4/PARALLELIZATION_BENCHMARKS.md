# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 06:16:18 UTC | hLk1Uxnw-iA_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 194.544 | 0.646 | 68.679 | 18.553 | 31.328 | 21.397 | 3.363 |

## 2026-06-26 06:16:18 UTC | hLk1Uxnw-iA_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/hLk1Uxnw-iA_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `194.544` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.646 |
| save_clips | - |
| sample_frames | 1.047 |
| caption_frames | 37.244 |
| sample_fps | 2.068 |
| detect_object_yolo | 8.770 |
| audio_scan | 11.959 |
| asr_timings | 29.123 |
| ast_timings | 27.588 |
| describe_scenes | 18.553 |
| summarize_scenes | 31.328 |
| synthesize_synopsis | 21.397 |
| make_embedding | 3.363 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.297 |
| branch_yolo_total | 10.844 |
| branch_audio_total | 68.679 |
