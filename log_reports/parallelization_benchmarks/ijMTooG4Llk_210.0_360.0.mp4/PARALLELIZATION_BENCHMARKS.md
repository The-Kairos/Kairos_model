# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 09:23:30 UTC | ijMTooG4Llk_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 138.110 | 0.805 | 46.387 | 12.374 | 13.156 | 19.135 | 2.835 |

## 2026-06-26 09:23:30 UTC | ijMTooG4Llk_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ijMTooG4Llk_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `138.110` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.805 |
| save_clips | - |
| sample_frames | 0.922 |
| caption_frames | 30.668 |
| sample_fps | 2.199 |
| detect_object_yolo | 8.231 |
| audio_scan | 14.008 |
| asr_timings | 10.990 |
| ast_timings | 21.380 |
| describe_scenes | 12.374 |
| summarize_scenes | 13.156 |
| synthesize_synopsis | 19.135 |
| make_embedding | 2.835 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 31.595 |
| branch_yolo_total | 10.436 |
| branch_audio_total | 46.387 |
