# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 15:24:25 UTC | QhHvx4n03MU_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 233.883 | 0.724 | 71.543 | 29.569 | 32.954 | 20.085 | 5.127 |

## 2026-06-25 15:24:25 UTC | QhHvx4n03MU_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/QhHvx4n03MU_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `233.883` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.724 |
| save_clips | - |
| sample_frames | 1.813 |
| caption_frames | 56.757 |
| sample_fps | 2.519 |
| detect_object_yolo | 11.310 |
| audio_scan | 15.763 |
| asr_timings | 14.189 |
| ast_timings | 41.582 |
| describe_scenes | 29.569 |
| summarize_scenes | 32.954 |
| synthesize_synopsis | 20.085 |
| make_embedding | 5.127 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 58.576 |
| branch_yolo_total | 13.835 |
| branch_audio_total | 71.543 |
