# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 01:38:41 UTC | uh2qGWfmESk_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 184.763 | 0.779 | 65.889 | 16.321 | 13.831 | 8.258 | 5.355 |

## 2026-06-27 01:38:41 UTC | uh2qGWfmESk_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/uh2qGWfmESk_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `184.763` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.779 |
| save_clips | - |
| sample_frames | 1.377 |
| caption_frames | 57.549 |
| sample_fps | 2.490 |
| detect_object_yolo | 11.495 |
| audio_scan | 14.018 |
| asr_timings | 9.310 |
| ast_timings | 42.552 |
| describe_scenes | 16.321 |
| summarize_scenes | 13.831 |
| synthesize_synopsis | 8.258 |
| make_embedding | 5.355 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 58.932 |
| branch_yolo_total | 13.992 |
| branch_audio_total | 65.889 |
