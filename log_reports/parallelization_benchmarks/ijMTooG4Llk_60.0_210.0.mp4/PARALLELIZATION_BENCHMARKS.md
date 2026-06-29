# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 09:29:57 UTC | ijMTooG4Llk_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 213.730 | 0.822 | 60.721 | 21.048 | 39.016 | 22.782 | 4.723 |

## 2026-06-26 09:29:57 UTC | ijMTooG4Llk_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ijMTooG4Llk_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `213.730` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.822 |
| save_clips | - |
| sample_frames | 1.460 |
| caption_frames | 49.672 |
| sample_fps | 2.398 |
| detect_object_yolo | 9.675 |
| audio_scan | 15.095 |
| asr_timings | 9.337 |
| ast_timings | 36.281 |
| describe_scenes | 21.048 |
| summarize_scenes | 39.016 |
| synthesize_synopsis | 22.782 |
| make_embedding | 4.723 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.138 |
| branch_yolo_total | 12.078 |
| branch_audio_total | 60.721 |
