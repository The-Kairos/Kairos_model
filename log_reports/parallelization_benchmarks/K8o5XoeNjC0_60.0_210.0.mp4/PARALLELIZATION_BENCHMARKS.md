# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 06:06:53 UTC | K8o5XoeNjC0_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 113.134 | 0.672 | 37.440 | 13.959 | 6.826 | 17.192 | 2.341 |

## 2026-06-25 06:06:53 UTC | K8o5XoeNjC0_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/K8o5XoeNjC0_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `113.134` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.672 |
| save_clips | - |
| sample_frames | 0.452 |
| caption_frames | 23.570 |
| sample_fps | 1.837 |
| detect_object_yolo | 7.372 |
| audio_scan | 11.906 |
| asr_timings | 9.668 |
| ast_timings | 15.857 |
| describe_scenes | 13.959 |
| summarize_scenes | 6.826 |
| synthesize_synopsis | 17.192 |
| make_embedding | 2.341 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 24.028 |
| branch_yolo_total | 9.215 |
| branch_audio_total | 37.440 |
