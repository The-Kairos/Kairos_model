# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 12:51:08 UTC | k4LLzwmwJS8_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 210.938 | 0.787 | 67.170 | 20.403 | 13.963 | 28.779 | 5.416 |

## 2026-06-26 12:51:08 UTC | k4LLzwmwJS8_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/k4LLzwmwJS8_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `210.938` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.787 |
| save_clips | - |
| sample_frames | 1.526 |
| caption_frames | 57.611 |
| sample_fps | 2.544 |
| detect_object_yolo | 11.277 |
| audio_scan | 11.899 |
| asr_timings | 10.912 |
| ast_timings | 44.350 |
| describe_scenes | 20.403 |
| summarize_scenes | 13.963 |
| synthesize_synopsis | 28.779 |
| make_embedding | 5.416 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 59.143 |
| branch_yolo_total | 13.827 |
| branch_audio_total | 67.170 |
