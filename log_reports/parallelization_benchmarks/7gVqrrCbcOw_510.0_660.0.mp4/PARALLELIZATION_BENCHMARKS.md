# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 16:31:55 UTC | 7gVqrrCbcOw_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 90.643 | 0.690 | 34.164 | 6.412 | 6.043 | 16.948 | 2.020 |

## 2026-06-24 16:31:55 UTC | 7gVqrrCbcOw_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/7gVqrrCbcOw_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `90.643` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.690 |
| save_clips | - |
| sample_frames | 0.287 |
| caption_frames | 15.486 |
| sample_fps | 1.641 |
| detect_object_yolo | 5.567 |
| audio_scan | 15.504 |
| asr_timings | 8.425 |
| ast_timings | 10.226 |
| describe_scenes | 6.412 |
| summarize_scenes | 6.043 |
| synthesize_synopsis | 16.948 |
| make_embedding | 2.020 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 15.779 |
| branch_yolo_total | 7.214 |
| branch_audio_total | 34.164 |
