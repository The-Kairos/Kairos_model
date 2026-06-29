# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 04:07:28 UTC | xqupGA-blZs_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 182.019 | 0.798 | 64.863 | 17.957 | 9.164 | 7.615 | 5.375 |

## 2026-06-27 04:07:28 UTC | xqupGA-blZs_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/xqupGA-blZs_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `182.019` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.798 |
| save_clips | - |
| sample_frames | 1.668 |
| caption_frames | 59.525 |
| sample_fps | 2.610 |
| detect_object_yolo | 11.029 |
| audio_scan | 9.766 |
| asr_timings | 10.239 |
| ast_timings | 44.850 |
| describe_scenes | 17.957 |
| summarize_scenes | 9.164 |
| synthesize_synopsis | 7.615 |
| make_embedding | 5.375 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 61.199 |
| branch_yolo_total | 13.645 |
| branch_audio_total | 64.863 |
