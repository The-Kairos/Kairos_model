# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 03:44:28 UTC | xGWnyxWo4sk_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 159.550 | 0.786 | 62.147 | 9.824 | 16.680 | 10.207 | 3.596 |

## 2026-06-27 03:44:28 UTC | xGWnyxWo4sk_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/xGWnyxWo4sk_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `159.550` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.786 |
| save_clips | - |
| sample_frames | 1.273 |
| caption_frames | 41.960 |
| sample_fps | 2.335 |
| detect_object_yolo | 9.298 |
| audio_scan | 15.240 |
| asr_timings | 16.593 |
| ast_timings | 30.306 |
| describe_scenes | 9.824 |
| summarize_scenes | 16.680 |
| synthesize_synopsis | 10.207 |
| make_embedding | 3.596 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.240 |
| branch_yolo_total | 11.639 |
| branch_audio_total | 62.147 |
