# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 04:19:41 UTC | xtmc4rgoxU4_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 169.210 | 0.785 | 69.631 | 13.914 | 7.826 | 6.173 | 4.661 |

## 2026-06-27 04:19:41 UTC | xtmc4rgoxU4_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/xtmc4rgoxU4_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `169.210` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.785 |
| save_clips | - |
| sample_frames | 1.278 |
| caption_frames | 50.790 |
| sample_fps | 2.452 |
| detect_object_yolo | 10.300 |
| audio_scan | 14.117 |
| asr_timings | 17.018 |
| ast_timings | 38.487 |
| describe_scenes | 13.914 |
| summarize_scenes | 7.826 |
| synthesize_synopsis | 6.173 |
| make_embedding | 4.661 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 52.074 |
| branch_yolo_total | 12.758 |
| branch_audio_total | 69.631 |
