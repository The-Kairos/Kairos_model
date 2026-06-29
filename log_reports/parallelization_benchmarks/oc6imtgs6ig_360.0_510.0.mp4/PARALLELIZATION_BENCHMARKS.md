# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 07:23:06 UTC | oc6imtgs6ig_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 153.662 | 0.820 | 54.707 | 11.553 | 5.776 | 16.228 | 4.170 |

## 2026-06-28 07:23:06 UTC | oc6imtgs6ig_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/oc6imtgs6ig_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `153.662` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.820 |
| save_clips | - |
| sample_frames | 1.412 |
| caption_frames | 45.718 |
| sample_fps | 2.408 |
| detect_object_yolo | 9.472 |
| audio_scan | 10.685 |
| asr_timings | 8.969 |
| ast_timings | 35.044 |
| describe_scenes | 11.553 |
| summarize_scenes | 5.776 |
| synthesize_synopsis | 16.228 |
| make_embedding | 4.170 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.136 |
| branch_yolo_total | 11.886 |
| branch_audio_total | 54.707 |
