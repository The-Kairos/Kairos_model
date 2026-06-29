# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 15:20:30 UTC | QhHvx4n03MU_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 138.665 | 0.718 | 41.327 | 15.333 | 8.466 | 36.480 | 2.328 |

## 2026-06-25 15:20:30 UTC | QhHvx4n03MU_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/QhHvx4n03MU_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `138.665` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.718 |
| save_clips | - |
| sample_frames | 0.631 |
| caption_frames | 22.816 |
| sample_fps | 1.915 |
| detect_object_yolo | 7.147 |
| audio_scan | 14.649 |
| asr_timings | 10.761 |
| ast_timings | 15.908 |
| describe_scenes | 15.333 |
| summarize_scenes | 8.466 |
| synthesize_synopsis | 36.480 |
| make_embedding | 2.328 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 23.453 |
| branch_yolo_total | 9.068 |
| branch_audio_total | 41.327 |
