# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 22:30:32 UTC | DNDMJEnD2oY_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 162.053 | 0.713 | 58.908 | 12.778 | 14.438 | 11.808 | 4.109 |

## 2026-06-24 22:30:32 UTC | DNDMJEnD2oY_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/DNDMJEnD2oY_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `162.053` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.713 |
| save_clips | - |
| sample_frames | 1.356 |
| caption_frames | 44.304 |
| sample_fps | 2.321 |
| detect_object_yolo | 9.864 |
| audio_scan | 13.146 |
| asr_timings | 9.910 |
| ast_timings | 35.844 |
| describe_scenes | 12.778 |
| summarize_scenes | 14.438 |
| synthesize_synopsis | 11.808 |
| make_embedding | 4.109 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.666 |
| branch_yolo_total | 12.191 |
| branch_audio_total | 58.908 |
