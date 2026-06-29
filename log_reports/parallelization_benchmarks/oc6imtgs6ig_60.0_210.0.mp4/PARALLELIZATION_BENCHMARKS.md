# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 07:25:26 UTC | oc6imtgs6ig_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 138.919 | 0.789 | 54.622 | 7.593 | 10.981 | 9.311 | 3.608 |

## 2026-06-28 07:25:26 UTC | oc6imtgs6ig_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/oc6imtgs6ig_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `138.919` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.789 |
| save_clips | - |
| sample_frames | 1.155 |
| caption_frames | 38.461 |
| sample_fps | 2.278 |
| detect_object_yolo | 8.738 |
| audio_scan | 13.810 |
| asr_timings | 11.160 |
| ast_timings | 29.644 |
| describe_scenes | 7.593 |
| summarize_scenes | 10.981 |
| synthesize_synopsis | 9.311 |
| make_embedding | 3.608 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.622 |
| branch_yolo_total | 11.022 |
| branch_audio_total | 54.622 |
