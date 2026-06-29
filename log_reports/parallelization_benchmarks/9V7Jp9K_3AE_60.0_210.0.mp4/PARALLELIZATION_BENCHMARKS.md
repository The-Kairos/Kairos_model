# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 18:04:27 UTC | 9V7Jp9K_3AE_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 202.112 | 0.668 | 90.168 | 16.684 | 9.414 | 26.274 | 3.826 |

## 2026-06-24 18:04:27 UTC | 9V7Jp9K_3AE_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/9V7Jp9K_3AE_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `202.112` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.668 |
| save_clips | - |
| sample_frames | 1.109 |
| caption_frames | 41.430 |
| sample_fps | 2.177 |
| detect_object_yolo | 8.936 |
| audio_scan | 10.758 |
| asr_timings | 46.784 |
| ast_timings | 32.618 |
| describe_scenes | 16.684 |
| summarize_scenes | 9.414 |
| synthesize_synopsis | 26.274 |
| make_embedding | 3.826 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 42.545 |
| branch_yolo_total | 11.119 |
| branch_audio_total | 90.168 |
