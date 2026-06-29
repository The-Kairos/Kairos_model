# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 08:06:32 UTC | -OmUBsxPguE_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 205.354 | 0.724 | 62.961 | 22.612 | 21.178 | 25.940 | 5.202 |

## 2026-06-24 08:06:32 UTC | -OmUBsxPguE_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-OmUBsxPguE_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `205.354` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.724 |
| save_clips | - |
| sample_frames | 1.839 |
| caption_frames | 50.633 |
| sample_fps | 2.509 |
| detect_object_yolo | 10.419 |
| audio_scan | 12.823 |
| asr_timings | 9.362 |
| ast_timings | 40.768 |
| describe_scenes | 22.612 |
| summarize_scenes | 21.178 |
| synthesize_synopsis | 25.940 |
| make_embedding | 5.202 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 52.478 |
| branch_yolo_total | 12.934 |
| branch_audio_total | 62.961 |
