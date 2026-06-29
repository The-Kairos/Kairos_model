# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 02:48:11 UTC | wFVkOVSNrEU_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 117.058 | 0.764 | 46.538 | 9.272 | 5.256 | 9.256 | 2.788 |

## 2026-06-27 02:48:11 UTC | wFVkOVSNrEU_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/wFVkOVSNrEU_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `117.058` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.764 |
| save_clips | - |
| sample_frames | 0.641 |
| caption_frames | 30.607 |
| sample_fps | 1.768 |
| detect_object_yolo | 8.765 |
| audio_scan | 15.117 |
| asr_timings | 10.248 |
| ast_timings | 21.164 |
| describe_scenes | 9.272 |
| summarize_scenes | 5.256 |
| synthesize_synopsis | 9.256 |
| make_embedding | 2.788 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 31.254 |
| branch_yolo_total | 10.539 |
| branch_audio_total | 46.538 |
