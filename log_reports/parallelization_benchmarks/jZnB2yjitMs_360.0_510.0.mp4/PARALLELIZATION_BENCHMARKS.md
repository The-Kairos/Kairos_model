# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 11:10:44 UTC | jZnB2yjitMs_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 208.311 | 0.844 | 69.439 | 16.998 | 20.520 | 22.396 | 5.040 |

## 2026-06-26 11:10:44 UTC | jZnB2yjitMs_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jZnB2yjitMs_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `208.311` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.844 |
| save_clips | - |
| sample_frames | 1.271 |
| caption_frames | 56.572 |
| sample_fps | 2.473 |
| detect_object_yolo | 11.253 |
| audio_scan | 16.342 |
| asr_timings | 11.640 |
| ast_timings | 41.447 |
| describe_scenes | 16.998 |
| summarize_scenes | 20.520 |
| synthesize_synopsis | 22.396 |
| make_embedding | 5.040 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 57.849 |
| branch_yolo_total | 13.732 |
| branch_audio_total | 69.439 |
