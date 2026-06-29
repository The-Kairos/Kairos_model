# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 16:45:32 UTC | 8ETILC-7U1w_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 163.887 | 0.681 | 53.857 | 13.918 | 15.704 | 24.517 | 3.650 |

## 2026-06-24 16:45:32 UTC | 8ETILC-7U1w_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/8ETILC-7U1w_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `163.887` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.681 |
| save_clips | - |
| sample_frames | 1.090 |
| caption_frames | 38.285 |
| sample_fps | 2.067 |
| detect_object_yolo | 8.735 |
| audio_scan | 14.864 |
| asr_timings | 10.148 |
| ast_timings | 28.837 |
| describe_scenes | 13.918 |
| summarize_scenes | 15.704 |
| synthesize_synopsis | 24.517 |
| make_embedding | 3.650 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.382 |
| branch_yolo_total | 10.808 |
| branch_audio_total | 53.857 |
