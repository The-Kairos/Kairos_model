# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 10:01:38 UTC | izWkhAfNNQg_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 167.334 | 0.812 | 57.307 | 16.901 | 9.788 | 21.653 | 4.196 |

## 2026-06-26 10:01:38 UTC | izWkhAfNNQg_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/izWkhAfNNQg_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `167.334` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.812 |
| save_clips | - |
| sample_frames | 1.242 |
| caption_frames | 42.557 |
| sample_fps | 2.332 |
| detect_object_yolo | 9.115 |
| audio_scan | 15.159 |
| asr_timings | 9.472 |
| ast_timings | 32.668 |
| describe_scenes | 16.901 |
| summarize_scenes | 9.788 |
| synthesize_synopsis | 21.653 |
| make_embedding | 4.196 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.805 |
| branch_yolo_total | 11.453 |
| branch_audio_total | 57.307 |
