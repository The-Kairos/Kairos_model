# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 10:45:26 UTC | jL7v3S2mFww_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 110.240 | 0.713 | 40.195 | 8.533 | 9.918 | 19.060 | 2.310 |

## 2026-06-26 10:45:26 UTC | jL7v3S2mFww_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jL7v3S2mFww_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `110.240` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.713 |
| save_clips | - |
| sample_frames | 0.536 |
| caption_frames | 18.872 |
| sample_fps | 1.865 |
| detect_object_yolo | 6.846 |
| audio_scan | 15.115 |
| asr_timings | 9.418 |
| ast_timings | 15.652 |
| describe_scenes | 8.533 |
| summarize_scenes | 9.918 |
| synthesize_synopsis | 19.060 |
| make_embedding | 2.310 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 19.414 |
| branch_yolo_total | 8.717 |
| branch_audio_total | 40.195 |
