# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 16:21:55 UTC | RFibtPMmIVU_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 146.318 | 0.794 | 58.792 | 15.620 | 8.462 | 14.077 | 3.082 |

## 2026-06-25 16:21:55 UTC | RFibtPMmIVU_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/RFibtPMmIVU_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `146.318` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.794 |
| save_clips | - |
| sample_frames | 0.833 |
| caption_frames | 33.223 |
| sample_fps | 2.126 |
| detect_object_yolo | 7.870 |
| audio_scan | 15.589 |
| asr_timings | 19.205 |
| ast_timings | 23.990 |
| describe_scenes | 15.620 |
| summarize_scenes | 8.462 |
| synthesize_synopsis | 14.077 |
| make_embedding | 3.082 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 34.062 |
| branch_yolo_total | 10.001 |
| branch_audio_total | 58.792 |
