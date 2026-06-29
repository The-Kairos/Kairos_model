# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 16:43:28 UTC | nkJy_9F8nS4_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 146.701 | 0.802 | 63.550 | 10.179 | 6.798 | 11.119 | 3.298 |

## 2026-06-27 16:43:28 UTC | nkJy_9F8nS4_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/nkJy_9F8nS4_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `146.701` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.802 |
| save_clips | - |
| sample_frames | 1.129 |
| caption_frames | 37.843 |
| sample_fps | 2.275 |
| detect_object_yolo | 8.326 |
| audio_scan | 13.756 |
| asr_timings | 22.837 |
| ast_timings | 26.949 |
| describe_scenes | 10.179 |
| summarize_scenes | 6.798 |
| synthesize_synopsis | 11.119 |
| make_embedding | 3.298 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.978 |
| branch_yolo_total | 10.607 |
| branch_audio_total | 63.550 |
