# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 22:22:04 UTC | 3r7kOP_nYNo_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 133.488 | 0.666 | 49.395 | 8.985 | 9.300 | 5.776 | 3.594 |

## 2026-06-21 22:22:04 UTC | 3r7kOP_nYNo_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3r7kOP_nYNo_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `133.488` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.666 |
| save_clips | - |
| sample_frames | 1.078 |
| caption_frames | 41.850 |
| sample_fps | 2.187 |
| detect_object_yolo | 9.247 |
| audio_scan | 11.830 |
| asr_timings | 7.420 |
| ast_timings | 30.136 |
| describe_scenes | 8.985 |
| summarize_scenes | 9.300 |
| synthesize_synopsis | 5.776 |
| make_embedding | 3.594 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 42.935 |
| branch_yolo_total | 11.440 |
| branch_audio_total | 49.395 |
