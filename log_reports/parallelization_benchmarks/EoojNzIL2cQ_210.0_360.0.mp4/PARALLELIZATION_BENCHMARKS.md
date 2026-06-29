# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 23:51:06 UTC | EoojNzIL2cQ_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 123.486 | 0.910 | 50.140 | 8.966 | 5.719 | 9.729 | 3.037 |

## 2026-06-24 23:51:06 UTC | EoojNzIL2cQ_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/EoojNzIL2cQ_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `123.486` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.910 |
| save_clips | - |
| sample_frames | 0.912 |
| caption_frames | 32.433 |
| sample_fps | 2.201 |
| detect_object_yolo | 8.052 |
| audio_scan | 15.123 |
| asr_timings | 11.101 |
| ast_timings | 23.908 |
| describe_scenes | 8.966 |
| summarize_scenes | 5.719 |
| synthesize_synopsis | 9.729 |
| make_embedding | 3.037 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 33.350 |
| branch_yolo_total | 10.259 |
| branch_audio_total | 50.140 |
