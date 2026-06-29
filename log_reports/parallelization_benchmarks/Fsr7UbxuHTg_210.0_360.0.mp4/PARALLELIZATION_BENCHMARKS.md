# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 00:41:12 UTC | Fsr7UbxuHTg_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 147.922 | 0.844 | 46.776 | 10.145 | 27.668 | 9.423 | 3.329 |

## 2026-06-25 00:41:12 UTC | Fsr7UbxuHTg_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Fsr7UbxuHTg_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `147.922` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.844 |
| save_clips | - |
| sample_frames | 1.111 |
| caption_frames | 36.277 |
| sample_fps | 2.230 |
| detect_object_yolo | 8.705 |
| audio_scan | 10.762 |
| asr_timings | 8.524 |
| ast_timings | 27.482 |
| describe_scenes | 10.145 |
| summarize_scenes | 27.668 |
| synthesize_synopsis | 9.423 |
| make_embedding | 3.329 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.393 |
| branch_yolo_total | 10.941 |
| branch_audio_total | 46.776 |
