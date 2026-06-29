# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 22:59:20 UTC | ZudB4C8rtQU_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 166.170 | 0.664 | 59.991 | 13.449 | 10.791 | 8.781 | 5.093 |

## 2026-06-25 22:59:20 UTC | ZudB4C8rtQU_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ZudB4C8rtQU_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `166.170` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.664 |
| save_clips | - |
| sample_frames | 1.349 |
| caption_frames | 52.201 |
| sample_fps | 2.360 |
| detect_object_yolo | 10.084 |
| audio_scan | 10.583 |
| asr_timings | 9.636 |
| ast_timings | 39.763 |
| describe_scenes | 13.449 |
| summarize_scenes | 10.791 |
| synthesize_synopsis | 8.781 |
| make_embedding | 5.093 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 53.556 |
| branch_yolo_total | 12.450 |
| branch_audio_total | 59.991 |
