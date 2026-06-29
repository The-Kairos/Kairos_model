# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 01:29:15 UTC | uh2qGWfmESk_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 154.112 | 0.828 | 56.421 | 13.591 | 9.217 | 9.091 | 4.118 |

## 2026-06-27 01:29:15 UTC | uh2qGWfmESk_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/uh2qGWfmESk_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `154.112` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.828 |
| save_clips | - |
| sample_frames | 1.157 |
| caption_frames | 46.035 |
| sample_fps | 2.367 |
| detect_object_yolo | 9.895 |
| audio_scan | 12.874 |
| asr_timings | 7.698 |
| ast_timings | 35.840 |
| describe_scenes | 13.591 |
| summarize_scenes | 9.217 |
| synthesize_synopsis | 9.091 |
| make_embedding | 4.118 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.198 |
| branch_yolo_total | 12.268 |
| branch_audio_total | 56.421 |
