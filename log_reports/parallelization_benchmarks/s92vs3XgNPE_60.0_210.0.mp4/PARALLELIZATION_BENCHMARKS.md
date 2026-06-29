# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 19:39:06 UTC | s92vs3XgNPE_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 149.048 | 0.774 | 62.331 | 12.867 | 6.990 | 13.142 | 3.321 |

## 2026-06-26 19:39:06 UTC | s92vs3XgNPE_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/s92vs3XgNPE_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `149.048` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.774 |
| save_clips | - |
| sample_frames | 0.960 |
| caption_frames | 36.125 |
| sample_fps | 2.182 |
| detect_object_yolo | 8.922 |
| audio_scan | 6.445 |
| asr_timings | 28.506 |
| ast_timings | 27.372 |
| describe_scenes | 12.867 |
| summarize_scenes | 6.990 |
| synthesize_synopsis | 13.142 |
| make_embedding | 3.321 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.091 |
| branch_yolo_total | 11.109 |
| branch_audio_total | 62.331 |
