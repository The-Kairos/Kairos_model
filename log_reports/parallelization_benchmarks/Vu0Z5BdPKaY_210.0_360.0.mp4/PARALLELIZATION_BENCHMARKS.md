# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 20:13:46 UTC | Vu0Z5BdPKaY_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 217.776 | 0.776 | 69.865 | 21.473 | 28.674 | 15.068 | 5.679 |

## 2026-06-25 20:13:46 UTC | Vu0Z5BdPKaY_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Vu0Z5BdPKaY_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `217.776` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.776 |
| save_clips | - |
| sample_frames | 1.299 |
| caption_frames | 59.399 |
| sample_fps | 2.454 |
| detect_object_yolo | 11.684 |
| audio_scan | 13.924 |
| asr_timings | 9.915 |
| ast_timings | 46.018 |
| describe_scenes | 21.473 |
| summarize_scenes | 28.674 |
| synthesize_synopsis | 15.068 |
| make_embedding | 5.679 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 60.704 |
| branch_yolo_total | 14.143 |
| branch_audio_total | 69.865 |
