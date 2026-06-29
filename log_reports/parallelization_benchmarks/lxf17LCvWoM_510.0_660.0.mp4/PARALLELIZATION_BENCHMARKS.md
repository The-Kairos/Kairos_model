# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 17:25:26 UTC | lxf17LCvWoM_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 231.851 | 0.778 | 78.999 | 29.327 | 22.449 | 14.168 | 6.100 |

## 2026-06-26 17:25:26 UTC | lxf17LCvWoM_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/lxf17LCvWoM_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `231.851` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.778 |
| save_clips | - |
| sample_frames | 1.479 |
| caption_frames | 62.321 |
| sample_fps | 2.529 |
| detect_object_yolo | 12.285 |
| audio_scan | 10.709 |
| asr_timings | 18.405 |
| ast_timings | 49.877 |
| describe_scenes | 29.327 |
| summarize_scenes | 22.449 |
| synthesize_synopsis | 14.168 |
| make_embedding | 6.100 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 63.806 |
| branch_yolo_total | 14.819 |
| branch_audio_total | 78.999 |
