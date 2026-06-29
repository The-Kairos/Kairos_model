# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 00:58:21 UTC | G-4tJ63X5vo_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 230.404 | 0.631 | 74.890 | 18.797 | 29.796 | 17.984 | 6.012 |

## 2026-06-25 00:58:21 UTC | G-4tJ63X5vo_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/G-4tJ63X5vo_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `230.404` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.631 |
| save_clips | - |
| sample_frames | 1.391 |
| caption_frames | 64.399 |
| sample_fps | 2.335 |
| detect_object_yolo | 12.689 |
| audio_scan | 14.273 |
| asr_timings | 10.103 |
| ast_timings | 50.506 |
| describe_scenes | 18.797 |
| summarize_scenes | 29.796 |
| synthesize_synopsis | 17.984 |
| make_embedding | 6.012 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 65.796 |
| branch_yolo_total | 15.030 |
| branch_audio_total | 74.890 |
