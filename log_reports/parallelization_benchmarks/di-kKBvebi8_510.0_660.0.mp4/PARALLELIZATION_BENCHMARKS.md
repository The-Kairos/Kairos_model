# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 03:21:50 UTC | di-kKBvebi8_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 173.847 | 0.803 | 59.827 | 14.617 | 22.947 | 10.081 | 4.387 |

## 2026-06-26 03:21:50 UTC | di-kKBvebi8_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/di-kKBvebi8_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `173.847` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.803 |
| save_clips | - |
| sample_frames | 1.673 |
| caption_frames | 46.271 |
| sample_fps | 2.490 |
| detect_object_yolo | 9.370 |
| audio_scan | 15.170 |
| asr_timings | 10.337 |
| ast_timings | 34.311 |
| describe_scenes | 14.617 |
| summarize_scenes | 22.947 |
| synthesize_synopsis | 10.081 |
| make_embedding | 4.387 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.950 |
| branch_yolo_total | 11.866 |
| branch_audio_total | 59.827 |
