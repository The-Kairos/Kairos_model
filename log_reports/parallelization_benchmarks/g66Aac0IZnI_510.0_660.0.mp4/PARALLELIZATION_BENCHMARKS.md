# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 04:55:07 UTC | g66Aac0IZnI_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 206.359 | 0.865 | 61.769 | 28.136 | 22.206 | 18.819 | 4.904 |

## 2026-06-26 04:55:07 UTC | g66Aac0IZnI_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/g66Aac0IZnI_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `206.359` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.865 |
| save_clips | - |
| sample_frames | 1.843 |
| caption_frames | 52.612 |
| sample_fps | 2.709 |
| detect_object_yolo | 10.989 |
| audio_scan | 14.265 |
| asr_timings | 8.048 |
| ast_timings | 39.447 |
| describe_scenes | 28.136 |
| summarize_scenes | 22.206 |
| synthesize_synopsis | 18.819 |
| make_embedding | 4.904 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 54.461 |
| branch_yolo_total | 13.703 |
| branch_audio_total | 61.769 |
