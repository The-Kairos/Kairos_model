# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 00:06:33 UTC | tvDH4JM_gME_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 197.553 | 0.785 | 66.121 | 20.537 | 16.157 | 8.050 | 5.644 |

## 2026-06-27 00:06:33 UTC | tvDH4JM_gME_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/tvDH4JM_gME_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `197.553` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.785 |
| save_clips | - |
| sample_frames | 1.696 |
| caption_frames | 63.034 |
| sample_fps | 2.606 |
| detect_object_yolo | 11.460 |
| audio_scan | 8.555 |
| asr_timings | 10.507 |
| ast_timings | 47.051 |
| describe_scenes | 20.537 |
| summarize_scenes | 16.157 |
| synthesize_synopsis | 8.050 |
| make_embedding | 5.644 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 64.736 |
| branch_yolo_total | 14.072 |
| branch_audio_total | 66.121 |
