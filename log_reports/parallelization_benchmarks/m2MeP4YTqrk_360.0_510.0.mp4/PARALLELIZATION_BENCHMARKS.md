# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 17:50:11 UTC | m2MeP4YTqrk_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 151.049 | 0.822 | 49.309 | 20.754 | 11.643 | 14.484 | 3.338 |

## 2026-06-26 17:50:11 UTC | m2MeP4YTqrk_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/m2MeP4YTqrk_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `151.049` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.822 |
| save_clips | - |
| sample_frames | 1.173 |
| caption_frames | 37.181 |
| sample_fps | 2.295 |
| detect_object_yolo | 8.609 |
| audio_scan | 14.144 |
| asr_timings | 7.303 |
| ast_timings | 27.853 |
| describe_scenes | 20.754 |
| summarize_scenes | 11.643 |
| synthesize_synopsis | 14.484 |
| make_embedding | 3.338 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.359 |
| branch_yolo_total | 10.911 |
| branch_audio_total | 49.309 |
