# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 13:18:36 UTC | 6xQqXvwyLbg_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 156.696 | 0.797 | 66.116 | 18.179 | 7.539 | 23.017 | 2.577 |

## 2026-06-24 13:18:36 UTC | 6xQqXvwyLbg_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/6xQqXvwyLbg_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `156.696` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.797 |
| save_clips | - |
| sample_frames | 0.981 |
| caption_frames | 26.663 |
| sample_fps | 2.097 |
| detect_object_yolo | 7.349 |
| audio_scan | 9.656 |
| asr_timings | 37.810 |
| ast_timings | 18.640 |
| describe_scenes | 18.179 |
| summarize_scenes | 7.539 |
| synthesize_synopsis | 23.017 |
| make_embedding | 2.577 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 27.649 |
| branch_yolo_total | 9.452 |
| branch_audio_total | 66.116 |
