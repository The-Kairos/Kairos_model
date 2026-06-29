# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 23:20:34 UTC | Dz0MY6ARnU4_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 109.401 | 1.313 | 73.643 | 3.824 | 3.106 | 8.992 | 1.271 |

## 2026-06-24 23:20:34 UTC | Dz0MY6ARnU4_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Dz0MY6ARnU4_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `109.401` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.313 |
| save_clips | - |
| sample_frames | 0.099 |
| caption_frames | 8.274 |
| sample_fps | 1.606 |
| detect_object_yolo | 5.873 |
| audio_scan | 16.077 |
| asr_timings | 52.793 |
| ast_timings | 4.763 |
| describe_scenes | 3.824 |
| summarize_scenes | 3.106 |
| synthesize_synopsis | 8.992 |
| make_embedding | 1.271 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 8.379 |
| branch_yolo_total | 7.485 |
| branch_audio_total | 73.643 |
