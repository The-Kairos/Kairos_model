# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 19:54:26 UTC | BwG_4D8xuLk_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 110.668 | 0.828 | 40.097 | 8.352 | 7.815 | 10.864 | 2.527 |

## 2026-06-24 19:54:26 UTC | BwG_4D8xuLk_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/BwG_4D8xuLk_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `110.668` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.828 |
| save_clips | - |
| sample_frames | 0.672 |
| caption_frames | 27.857 |
| sample_fps | 2.087 |
| detect_object_yolo | 8.115 |
| audio_scan | 10.845 |
| asr_timings | 10.409 |
| ast_timings | 18.834 |
| describe_scenes | 8.352 |
| summarize_scenes | 7.815 |
| synthesize_synopsis | 10.864 |
| make_embedding | 2.527 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 28.535 |
| branch_yolo_total | 10.208 |
| branch_audio_total | 40.097 |
