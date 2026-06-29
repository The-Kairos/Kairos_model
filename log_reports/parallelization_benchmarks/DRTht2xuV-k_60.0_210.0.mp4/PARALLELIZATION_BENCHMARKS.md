# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 22:51:46 UTC | DRTht2xuV-k_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 219.634 | 0.792 | 109.100 | 15.835 | 17.239 | 7.679 | 4.470 |

## 2026-06-24 22:51:46 UTC | DRTht2xuV-k_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/DRTht2xuV-k_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `219.634` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.792 |
| save_clips | - |
| sample_frames | 1.256 |
| caption_frames | 49.392 |
| sample_fps | 2.383 |
| detect_object_yolo | 10.083 |
| audio_scan | 15.005 |
| asr_timings | 57.341 |
| ast_timings | 36.745 |
| describe_scenes | 15.835 |
| summarize_scenes | 17.239 |
| synthesize_synopsis | 7.679 |
| make_embedding | 4.470 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 50.654 |
| branch_yolo_total | 12.472 |
| branch_audio_total | 109.100 |
