# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 03:35:59 UTC | HQBXQyT8UoI_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 191.536 | 0.819 | 66.000 | 12.609 | 13.287 | 15.135 | 5.654 |

## 2026-06-25 03:35:59 UTC | HQBXQyT8UoI_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/HQBXQyT8UoI_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `191.536` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.819 |
| save_clips | - |
| sample_frames | 1.676 |
| caption_frames | 60.059 |
| sample_fps | 2.655 |
| detect_object_yolo | 12.174 |
| audio_scan | 10.685 |
| asr_timings | 9.243 |
| ast_timings | 46.064 |
| describe_scenes | 12.609 |
| summarize_scenes | 13.287 |
| synthesize_synopsis | 15.135 |
| make_embedding | 5.654 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 61.740 |
| branch_yolo_total | 14.834 |
| branch_audio_total | 66.000 |
