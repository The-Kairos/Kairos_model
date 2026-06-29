# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 16:41:50 UTC | ljcGQQxxO1E_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 1956.810 | 0.659 | 1839.129 | 23.052 | 12.869 | 22.227 | 3.603 |

## 2026-06-26 16:41:50 UTC | ljcGQQxxO1E_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ljcGQQxxO1E_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `1956.810` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.659 |
| save_clips | - |
| sample_frames | 1.033 |
| caption_frames | 42.090 |
| sample_fps | 2.061 |
| detect_object_yolo | 8.620 |
| audio_scan | 8.771 |
| asr_timings | 1800.412 |
| ast_timings | 29.937 |
| describe_scenes | 23.052 |
| summarize_scenes | 12.869 |
| synthesize_synopsis | 22.227 |
| make_embedding | 3.603 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.129 |
| branch_yolo_total | 10.687 |
| branch_audio_total | 1839.129 |
