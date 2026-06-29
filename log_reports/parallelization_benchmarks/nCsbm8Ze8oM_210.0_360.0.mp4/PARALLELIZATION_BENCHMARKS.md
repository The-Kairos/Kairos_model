# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 16:27:30 UTC | nCsbm8Ze8oM_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 148.711 | 0.859 | 47.912 | 10.714 | 20.855 | 13.857 | 3.540 |

## 2026-06-27 16:27:30 UTC | nCsbm8Ze8oM_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/nCsbm8Ze8oM_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `148.711` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.859 |
| save_clips | - |
| sample_frames | 0.975 |
| caption_frames | 37.957 |
| sample_fps | 2.265 |
| detect_object_yolo | 8.379 |
| audio_scan | 10.652 |
| asr_timings | 7.865 |
| ast_timings | 29.386 |
| describe_scenes | 10.714 |
| summarize_scenes | 20.855 |
| synthesize_synopsis | 13.857 |
| make_embedding | 3.540 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.938 |
| branch_yolo_total | 10.650 |
| branch_audio_total | 47.912 |
