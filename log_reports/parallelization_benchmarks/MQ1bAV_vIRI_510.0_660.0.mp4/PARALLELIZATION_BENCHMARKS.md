# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 09:33:34 UTC | MQ1bAV_vIRI_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 162.008 | 0.805 | 44.414 | 15.290 | 32.902 | 22.740 | 2.803 |

## 2026-06-25 09:33:34 UTC | MQ1bAV_vIRI_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/MQ1bAV_vIRI_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `162.008` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.805 |
| save_clips | - |
| sample_frames | 0.687 |
| caption_frames | 31.424 |
| sample_fps | 2.077 |
| detect_object_yolo | 7.428 |
| audio_scan | 12.834 |
| asr_timings | 9.582 |
| ast_timings | 21.990 |
| describe_scenes | 15.290 |
| summarize_scenes | 32.902 |
| synthesize_synopsis | 22.740 |
| make_embedding | 2.803 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.117 |
| branch_yolo_total | 9.511 |
| branch_audio_total | 44.414 |
