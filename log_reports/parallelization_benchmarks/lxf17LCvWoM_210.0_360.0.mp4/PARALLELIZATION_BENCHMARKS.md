# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 17:17:46 UTC | lxf17LCvWoM_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 180.551 | 0.768 | 56.926 | 37.997 | 15.057 | 16.930 | 3.617 |

## 2026-06-26 17:17:46 UTC | lxf17LCvWoM_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/lxf17LCvWoM_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `180.551` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.768 |
| save_clips | - |
| sample_frames | 1.037 |
| caption_frames | 35.197 |
| sample_fps | 2.305 |
| detect_object_yolo | 9.320 |
| audio_scan | 15.061 |
| asr_timings | 12.048 |
| ast_timings | 29.809 |
| describe_scenes | 37.997 |
| summarize_scenes | 15.057 |
| synthesize_synopsis | 16.930 |
| make_embedding | 3.617 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.240 |
| branch_yolo_total | 11.631 |
| branch_audio_total | 56.926 |
