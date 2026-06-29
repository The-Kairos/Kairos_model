# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 12:21:35 UTC | 6f_loPJsPFE_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 191.700 | 0.700 | 62.124 | 14.788 | 25.170 | 20.239 | 5.002 |

## 2026-06-24 12:21:35 UTC | 6f_loPJsPFE_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/6f_loPJsPFE_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `191.700` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.700 |
| save_clips | - |
| sample_frames | 1.555 |
| caption_frames | 48.201 |
| sample_fps | 2.386 |
| detect_object_yolo | 10.135 |
| audio_scan | 12.694 |
| asr_timings | 11.236 |
| ast_timings | 38.185 |
| describe_scenes | 14.788 |
| summarize_scenes | 25.170 |
| synthesize_synopsis | 20.239 |
| make_embedding | 5.002 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.762 |
| branch_yolo_total | 12.527 |
| branch_audio_total | 62.124 |
