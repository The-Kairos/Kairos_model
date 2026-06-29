# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 20:11:13 UTC | C9JnjCVIW5k_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 150.735 | 0.686 | 47.924 | 15.306 | 9.338 | 15.199 | 3.824 |

## 2026-06-24 20:11:13 UTC | C9JnjCVIW5k_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/C9JnjCVIW5k_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `150.735` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.686 |
| save_clips | - |
| sample_frames | 1.283 |
| caption_frames | 44.829 |
| sample_fps | 2.144 |
| detect_object_yolo | 8.790 |
| audio_scan | 9.063 |
| asr_timings | 5.948 |
| ast_timings | 32.904 |
| describe_scenes | 15.306 |
| summarize_scenes | 9.338 |
| synthesize_synopsis | 15.199 |
| make_embedding | 3.824 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.118 |
| branch_yolo_total | 10.939 |
| branch_audio_total | 47.924 |
