# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 16:09:29 UTC | QxP1p3EzOks_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 167.049 | 0.679 | 58.332 | 21.574 | 13.507 | 17.605 | 3.627 |

## 2026-06-25 16:09:29 UTC | QxP1p3EzOks_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/QxP1p3EzOks_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `167.049` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.679 |
| save_clips | - |
| sample_frames | 0.939 |
| caption_frames | 38.536 |
| sample_fps | 2.111 |
| detect_object_yolo | 8.700 |
| audio_scan | 15.549 |
| asr_timings | 12.746 |
| ast_timings | 30.029 |
| describe_scenes | 21.574 |
| summarize_scenes | 13.507 |
| synthesize_synopsis | 17.605 |
| make_embedding | 3.627 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.482 |
| branch_yolo_total | 10.817 |
| branch_audio_total | 58.332 |
