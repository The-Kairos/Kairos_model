# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 13:07:06 UTC | -ruy-w0bxvA_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 139.153 | 0.769 | 52.128 | 8.335 | 11.051 | 10.950 | 3.642 |

## 2026-06-27 13:07:06 UTC | -ruy-w0bxvA_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-ruy-w0bxvA_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `139.153` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.769 |
| save_clips | - |
| sample_frames | 0.900 |
| caption_frames | 38.986 |
| sample_fps | 2.232 |
| detect_object_yolo | 8.759 |
| audio_scan | 13.889 |
| asr_timings | 8.757 |
| ast_timings | 29.473 |
| describe_scenes | 8.335 |
| summarize_scenes | 11.051 |
| synthesize_synopsis | 10.950 |
| make_embedding | 3.642 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.892 |
| branch_yolo_total | 10.997 |
| branch_audio_total | 52.128 |
