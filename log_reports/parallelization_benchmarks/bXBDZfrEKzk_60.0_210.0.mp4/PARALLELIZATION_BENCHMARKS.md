# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 01:27:14 UTC | bXBDZfrEKzk_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 136.195 | 0.871 | 49.519 | 11.230 | 10.934 | 8.705 | 3.680 |

## 2026-06-26 01:27:14 UTC | bXBDZfrEKzk_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/bXBDZfrEKzk_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `136.195` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.871 |
| save_clips | - |
| sample_frames | 1.301 |
| caption_frames | 36.999 |
| sample_fps | 2.360 |
| detect_object_yolo | 9.198 |
| audio_scan | 10.800 |
| asr_timings | 8.463 |
| ast_timings | 30.247 |
| describe_scenes | 11.230 |
| summarize_scenes | 10.934 |
| synthesize_synopsis | 8.705 |
| make_embedding | 3.680 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.307 |
| branch_yolo_total | 11.563 |
| branch_audio_total | 49.519 |
