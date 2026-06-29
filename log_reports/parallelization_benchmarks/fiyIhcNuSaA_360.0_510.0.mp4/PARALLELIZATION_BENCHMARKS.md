# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 04:34:02 UTC | fiyIhcNuSaA_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 155.776 | 0.790 | 54.855 | 8.706 | 17.723 | 18.890 | 3.721 |

## 2026-06-26 04:34:02 UTC | fiyIhcNuSaA_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/fiyIhcNuSaA_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `155.776` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.790 |
| save_clips | - |
| sample_frames | 1.111 |
| caption_frames | 37.651 |
| sample_fps | 2.248 |
| detect_object_yolo | 8.681 |
| audio_scan | 15.089 |
| asr_timings | 10.120 |
| ast_timings | 29.638 |
| describe_scenes | 8.706 |
| summarize_scenes | 17.723 |
| synthesize_synopsis | 18.890 |
| make_embedding | 3.721 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.768 |
| branch_yolo_total | 10.934 |
| branch_audio_total | 54.855 |
