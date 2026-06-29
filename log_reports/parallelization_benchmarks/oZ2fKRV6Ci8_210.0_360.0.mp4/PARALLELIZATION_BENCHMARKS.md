# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 17:14:23 UTC | oZ2fKRV6Ci8_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 214.914 | 0.805 | 76.283 | 16.990 | 15.801 | 10.103 | 6.454 |

## 2026-06-27 17:14:23 UTC | oZ2fKRV6Ci8_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/oZ2fKRV6Ci8_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `214.914` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.805 |
| save_clips | - |
| sample_frames | 1.681 |
| caption_frames | 70.407 |
| sample_fps | 2.739 |
| detect_object_yolo | 12.258 |
| audio_scan | 15.914 |
| asr_timings | 9.348 |
| ast_timings | 51.012 |
| describe_scenes | 16.990 |
| summarize_scenes | 15.801 |
| synthesize_synopsis | 10.103 |
| make_embedding | 6.454 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 72.093 |
| branch_yolo_total | 15.003 |
| branch_audio_total | 76.283 |
