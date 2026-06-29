# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 22:40:26 UTC | DOET406zX8A_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 103.052 | 0.634 | 34.521 | 6.013 | 5.310 | 12.591 | 2.757 |

## 2026-06-24 22:40:26 UTC | DOET406zX8A_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/DOET406zX8A_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `103.052` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.634 |
| save_clips | - |
| sample_frames | 0.637 |
| caption_frames | 29.698 |
| sample_fps | 1.873 |
| detect_object_yolo | 7.600 |
| audio_scan | 5.420 |
| asr_timings | 16.349 |
| ast_timings | 12.744 |
| describe_scenes | 6.013 |
| summarize_scenes | 5.310 |
| synthesize_synopsis | 12.591 |
| make_embedding | 2.757 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.341 |
| branch_yolo_total | 9.479 |
| branch_audio_total | 34.521 |
