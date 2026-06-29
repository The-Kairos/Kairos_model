# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 17:42:24 UTC | 9Kt7THRXaJM_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 154.699 | 0.805 | 50.819 | 12.675 | 10.364 | 26.159 | 3.283 |

## 2026-06-24 17:42:24 UTC | 9Kt7THRXaJM_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/9Kt7THRXaJM_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `154.699` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.805 |
| save_clips | - |
| sample_frames | 0.994 |
| caption_frames | 37.870 |
| sample_fps | 2.223 |
| detect_object_yolo | 8.122 |
| audio_scan | 8.634 |
| asr_timings | 15.648 |
| ast_timings | 26.529 |
| describe_scenes | 12.675 |
| summarize_scenes | 10.364 |
| synthesize_synopsis | 26.159 |
| make_embedding | 3.283 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.870 |
| branch_yolo_total | 10.350 |
| branch_audio_total | 50.819 |
