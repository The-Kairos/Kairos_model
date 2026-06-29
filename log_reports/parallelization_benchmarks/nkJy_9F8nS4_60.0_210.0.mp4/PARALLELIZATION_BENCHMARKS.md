# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 16:48:25 UTC | nkJy_9F8nS4_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 165.886 | 0.818 | 61.727 | 12.302 | 11.067 | 7.432 | 4.465 |

## 2026-06-27 16:48:25 UTC | nkJy_9F8nS4_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/nkJy_9F8nS4_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `165.886` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.818 |
| save_clips | - |
| sample_frames | 1.488 |
| caption_frames | 52.145 |
| sample_fps | 2.463 |
| detect_object_yolo | 10.490 |
| audio_scan | 14.915 |
| asr_timings | 8.760 |
| ast_timings | 38.043 |
| describe_scenes | 12.302 |
| summarize_scenes | 11.067 |
| synthesize_synopsis | 7.432 |
| make_embedding | 4.465 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 53.640 |
| branch_yolo_total | 12.960 |
| branch_audio_total | 61.727 |
