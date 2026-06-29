# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 02:35:09 UTC | vsykXfqk_4A_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 216.572 | 0.773 | 85.097 | 19.432 | 9.850 | 8.189 | 6.061 |

## 2026-06-27 02:35:09 UTC | vsykXfqk_4A_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/vsykXfqk_4A_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `216.572` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.773 |
| save_clips | - |
| sample_frames | 1.941 |
| caption_frames | 68.358 |
| sample_fps | 2.788 |
| detect_object_yolo | 12.633 |
| audio_scan | 5.497 |
| asr_timings | 28.539 |
| ast_timings | 51.053 |
| describe_scenes | 19.432 |
| summarize_scenes | 9.850 |
| synthesize_synopsis | 8.189 |
| make_embedding | 6.061 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 70.304 |
| branch_yolo_total | 15.427 |
| branch_audio_total | 85.097 |
