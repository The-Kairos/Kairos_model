# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 13:38:27 UTC | kENu3bYxd5E_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 223.506 | 0.796 | 61.897 | 31.516 | 33.616 | 31.599 | 4.170 |

## 2026-06-26 13:38:27 UTC | kENu3bYxd5E_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/kENu3bYxd5E_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `223.506` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.796 |
| save_clips | - |
| sample_frames | 1.257 |
| caption_frames | 45.144 |
| sample_fps | 2.367 |
| detect_object_yolo | 9.701 |
| audio_scan | 16.096 |
| asr_timings | 10.594 |
| ast_timings | 35.199 |
| describe_scenes | 31.516 |
| summarize_scenes | 33.616 |
| synthesize_synopsis | 31.599 |
| make_embedding | 4.170 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.407 |
| branch_yolo_total | 12.074 |
| branch_audio_total | 61.897 |
