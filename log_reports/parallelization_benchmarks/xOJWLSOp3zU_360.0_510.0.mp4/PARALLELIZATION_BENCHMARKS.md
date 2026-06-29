# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 03:51:14 UTC | xOJWLSOp3zU_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 135.859 | 0.787 | 49.129 | 11.565 | 9.370 | 8.005 | 3.612 |

## 2026-06-27 03:51:14 UTC | xOJWLSOp3zU_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/xOJWLSOp3zU_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `135.859` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.787 |
| save_clips | - |
| sample_frames | 1.250 |
| caption_frames | 39.648 |
| sample_fps | 2.318 |
| detect_object_yolo | 8.738 |
| audio_scan | 10.862 |
| asr_timings | 7.996 |
| ast_timings | 30.263 |
| describe_scenes | 11.565 |
| summarize_scenes | 9.370 |
| synthesize_synopsis | 8.005 |
| make_embedding | 3.612 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 40.904 |
| branch_yolo_total | 11.062 |
| branch_audio_total | 49.129 |
