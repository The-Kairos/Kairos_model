# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 13:40:11 UTC | POU_6XcdD1s_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 192.947 | 0.625 | 74.970 | 23.979 | 39.473 | 18.590 | 2.335 |

## 2026-06-25 13:40:11 UTC | POU_6XcdD1s_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/POU_6XcdD1s_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `192.947` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.625 |
| save_clips | - |
| sample_frames | 0.599 |
| caption_frames | 22.160 |
| sample_fps | 1.841 |
| detect_object_yolo | 6.960 |
| audio_scan | 15.319 |
| asr_timings | 43.951 |
| ast_timings | 15.692 |
| describe_scenes | 23.979 |
| summarize_scenes | 39.473 |
| synthesize_synopsis | 18.590 |
| make_embedding | 2.335 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 22.765 |
| branch_yolo_total | 8.806 |
| branch_audio_total | 74.970 |
