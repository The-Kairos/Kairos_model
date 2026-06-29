# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 22:13:53 UTC | 3fESWnyZC0o_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 115.160 | 0.788 | 44.334 | 6.972 | 10.509 | 11.450 | 2.572 |

## 2026-06-21 22:13:53 UTC | 3fESWnyZC0o_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3fESWnyZC0o_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `115.160` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.788 |
| save_clips | - |
| sample_frames | 0.783 |
| caption_frames | 26.610 |
| sample_fps | 2.073 |
| detect_object_yolo | 7.657 |
| audio_scan | 15.985 |
| asr_timings | 9.839 |
| ast_timings | 18.501 |
| describe_scenes | 6.972 |
| summarize_scenes | 10.509 |
| synthesize_synopsis | 11.450 |
| make_embedding | 2.572 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 27.399 |
| branch_yolo_total | 9.736 |
| branch_audio_total | 44.334 |
