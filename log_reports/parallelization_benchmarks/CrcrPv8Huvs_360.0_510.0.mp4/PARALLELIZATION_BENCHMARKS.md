# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 20:59:19 UTC | CrcrPv8Huvs_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 138.215 | 0.713 | 46.524 | 11.733 | 7.227 | 24.922 | 3.059 |

## 2026-06-24 20:59:19 UTC | CrcrPv8Huvs_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/CrcrPv8Huvs_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `138.215` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.713 |
| save_clips | - |
| sample_frames | 1.175 |
| caption_frames | 31.240 |
| sample_fps | 2.082 |
| detect_object_yolo | 8.136 |
| audio_scan | 14.949 |
| asr_timings | 8.918 |
| ast_timings | 22.648 |
| describe_scenes | 11.733 |
| summarize_scenes | 7.227 |
| synthesize_synopsis | 24.922 |
| make_embedding | 3.059 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.421 |
| branch_yolo_total | 10.223 |
| branch_audio_total | 46.524 |
