# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 06:40:12 UTC | KUs4hm2MQ8o_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 132.428 | 0.637 | 45.331 | 10.407 | 10.682 | 18.255 | 3.030 |

## 2026-06-25 06:40:12 UTC | KUs4hm2MQ8o_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/KUs4hm2MQ8o_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `132.428` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.637 |
| save_clips | - |
| sample_frames | 0.686 |
| caption_frames | 32.368 |
| sample_fps | 1.914 |
| detect_object_yolo | 7.744 |
| audio_scan | 9.575 |
| asr_timings | 11.931 |
| ast_timings | 23.815 |
| describe_scenes | 10.407 |
| summarize_scenes | 10.682 |
| synthesize_synopsis | 18.255 |
| make_embedding | 3.030 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 33.059 |
| branch_yolo_total | 9.664 |
| branch_audio_total | 45.331 |
