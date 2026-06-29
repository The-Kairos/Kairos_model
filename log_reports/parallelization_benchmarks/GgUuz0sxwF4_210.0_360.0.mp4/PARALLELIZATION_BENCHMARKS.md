# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 01:59:16 UTC | GgUuz0sxwF4_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 147.531 | 0.814 | 62.883 | 7.732 | 13.680 | 16.141 | 3.024 |

## 2026-06-25 01:59:16 UTC | GgUuz0sxwF4_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/GgUuz0sxwF4_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `147.531` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.814 |
| save_clips | - |
| sample_frames | 0.860 |
| caption_frames | 30.799 |
| sample_fps | 2.142 |
| detect_object_yolo | 8.064 |
| audio_scan | 14.948 |
| asr_timings | 24.176 |
| ast_timings | 23.751 |
| describe_scenes | 7.732 |
| summarize_scenes | 13.680 |
| synthesize_synopsis | 16.141 |
| make_embedding | 3.024 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 31.664 |
| branch_yolo_total | 10.212 |
| branch_audio_total | 62.883 |
