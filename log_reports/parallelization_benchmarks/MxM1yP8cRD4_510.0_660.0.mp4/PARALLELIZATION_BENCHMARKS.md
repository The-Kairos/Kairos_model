# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 10:15:26 UTC | MxM1yP8cRD4_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 165.021 | 0.809 | 52.199 | 13.781 | 26.056 | 28.540 | 2.578 |

## 2026-06-25 10:15:26 UTC | MxM1yP8cRD4_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/MxM1yP8cRD4_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `165.021` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.809 |
| save_clips | - |
| sample_frames | 0.740 |
| caption_frames | 29.344 |
| sample_fps | 2.032 |
| detect_object_yolo | 7.533 |
| audio_scan | 14.842 |
| asr_timings | 18.586 |
| ast_timings | 18.763 |
| describe_scenes | 13.781 |
| summarize_scenes | 26.056 |
| synthesize_synopsis | 28.540 |
| make_embedding | 2.578 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.090 |
| branch_yolo_total | 9.572 |
| branch_audio_total | 52.199 |
