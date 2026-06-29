# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 04:57:53 UTC | yPyNZtMGZTk_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 175.106 | 0.658 | 59.947 | 12.151 | 24.889 | 6.024 | 4.645 |

## 2026-06-27 04:57:53 UTC | yPyNZtMGZTk_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/yPyNZtMGZTk_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `175.106` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.658 |
| save_clips | - |
| sample_frames | 1.265 |
| caption_frames | 51.325 |
| sample_fps | 2.315 |
| detect_object_yolo | 10.485 |
| audio_scan | 10.768 |
| asr_timings | 10.334 |
| ast_timings | 38.837 |
| describe_scenes | 12.151 |
| summarize_scenes | 24.889 |
| synthesize_synopsis | 6.024 |
| make_embedding | 4.645 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 52.596 |
| branch_yolo_total | 12.806 |
| branch_audio_total | 59.947 |
