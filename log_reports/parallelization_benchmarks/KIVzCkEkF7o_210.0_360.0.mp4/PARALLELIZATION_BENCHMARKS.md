# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 06:19:40 UTC | KIVzCkEkF7o_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 187.254 | 0.802 | 60.446 | 13.112 | 28.486 | 17.616 | 4.191 |

## 2026-06-25 06:19:40 UTC | KIVzCkEkF7o_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/KIVzCkEkF7o_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `187.254` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.802 |
| save_clips | - |
| sample_frames | 1.343 |
| caption_frames | 47.585 |
| sample_fps | 2.391 |
| detect_object_yolo | 9.877 |
| audio_scan | 13.811 |
| asr_timings | 11.907 |
| ast_timings | 34.720 |
| describe_scenes | 13.112 |
| summarize_scenes | 28.486 |
| synthesize_synopsis | 17.616 |
| make_embedding | 4.191 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 48.933 |
| branch_yolo_total | 12.274 |
| branch_audio_total | 60.446 |
