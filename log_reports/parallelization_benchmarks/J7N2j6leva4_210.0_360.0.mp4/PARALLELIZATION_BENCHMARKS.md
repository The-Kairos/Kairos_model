# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 04:58:59 UTC | J7N2j6leva4_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 135.949 | 0.664 | 48.429 | 12.426 | 10.618 | 15.172 | 3.011 |

## 2026-06-25 04:58:59 UTC | J7N2j6leva4_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/J7N2j6leva4_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `135.949` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.664 |
| save_clips | - |
| sample_frames | 0.904 |
| caption_frames | 33.003 |
| sample_fps | 1.997 |
| detect_object_yolo | 8.348 |
| audio_scan | 12.808 |
| asr_timings | 11.266 |
| ast_timings | 24.346 |
| describe_scenes | 12.426 |
| summarize_scenes | 10.618 |
| synthesize_synopsis | 15.172 |
| make_embedding | 3.011 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 33.913 |
| branch_yolo_total | 10.351 |
| branch_audio_total | 48.429 |
