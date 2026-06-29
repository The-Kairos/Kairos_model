# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 03:12:04 UTC | HAttn-5lM3Y_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 169.541 | 0.676 | 56.538 | 13.638 | 24.773 | 13.620 | 3.932 |

## 2026-06-25 03:12:04 UTC | HAttn-5lM3Y_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/HAttn-5lM3Y_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `169.541` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.676 |
| save_clips | - |
| sample_frames | 1.144 |
| caption_frames | 42.545 |
| sample_fps | 2.191 |
| detect_object_yolo | 9.101 |
| audio_scan | 15.833 |
| asr_timings | 9.577 |
| ast_timings | 31.120 |
| describe_scenes | 13.638 |
| summarize_scenes | 24.773 |
| synthesize_synopsis | 13.620 |
| make_embedding | 3.932 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.695 |
| branch_yolo_total | 11.297 |
| branch_audio_total | 56.538 |
