# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 13:40:19 UTC | 7JUPCETmSR8_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 194.525 | 0.684 | 56.480 | 30.099 | 12.946 | 25.487 | 4.254 |

## 2026-06-24 13:40:19 UTC | 7JUPCETmSR8_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/7JUPCETmSR8_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `194.525` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.684 |
| save_clips | - |
| sample_frames | 1.335 |
| caption_frames | 49.847 |
| sample_fps | 2.252 |
| detect_object_yolo | 9.717 |
| audio_scan | 11.766 |
| asr_timings | 8.627 |
| ast_timings | 36.077 |
| describe_scenes | 30.099 |
| summarize_scenes | 12.946 |
| synthesize_synopsis | 25.487 |
| make_embedding | 4.254 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.188 |
| branch_yolo_total | 11.975 |
| branch_audio_total | 56.480 |
