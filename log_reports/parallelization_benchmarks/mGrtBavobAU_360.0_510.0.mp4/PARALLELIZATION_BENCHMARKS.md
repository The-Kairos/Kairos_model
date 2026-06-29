# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 18:11:57 UTC | mGrtBavobAU_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 152.254 | 0.820 | 49.127 | 22.057 | 13.290 | 16.218 | 3.283 |

## 2026-06-26 18:11:57 UTC | mGrtBavobAU_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/mGrtBavobAU_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `152.254` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.820 |
| save_clips | - |
| sample_frames | 1.049 |
| caption_frames | 34.378 |
| sample_fps | 2.241 |
| detect_object_yolo | 8.337 |
| audio_scan | 13.842 |
| asr_timings | 8.336 |
| ast_timings | 26.941 |
| describe_scenes | 22.057 |
| summarize_scenes | 13.290 |
| synthesize_synopsis | 16.218 |
| make_embedding | 3.283 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 35.433 |
| branch_yolo_total | 10.583 |
| branch_audio_total | 49.127 |
