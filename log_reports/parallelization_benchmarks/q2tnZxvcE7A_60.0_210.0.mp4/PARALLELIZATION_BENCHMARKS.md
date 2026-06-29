# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 08:29:16 UTC | q2tnZxvcE7A_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 202.330 | 0.714 | 100.424 | 14.150 | 10.638 | 7.152 | 4.435 |

## 2026-06-28 08:29:16 UTC | q2tnZxvcE7A_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/q2tnZxvcE7A_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `202.330` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.714 |
| save_clips | - |
| sample_frames | 1.964 |
| caption_frames | 49.457 |
| sample_fps | 2.598 |
| detect_object_yolo | 9.412 |
| audio_scan | 13.836 |
| asr_timings | 47.943 |
| ast_timings | 38.637 |
| describe_scenes | 14.150 |
| summarize_scenes | 10.638 |
| synthesize_synopsis | 7.152 |
| make_embedding | 4.435 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.427 |
| branch_yolo_total | 12.016 |
| branch_audio_total | 100.424 |
