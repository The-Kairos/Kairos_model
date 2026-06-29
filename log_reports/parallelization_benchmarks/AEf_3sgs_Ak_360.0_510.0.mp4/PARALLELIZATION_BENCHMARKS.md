# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 18:39:04 UTC | AEf_3sgs_Ak_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 106.371 | 0.715 | 37.198 | 6.599 | 12.527 | 20.089 | 2.087 |

## 2026-06-24 18:39:04 UTC | AEf_3sgs_Ak_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/AEf_3sgs_Ak_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `106.371` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.715 |
| save_clips | - |
| sample_frames | 0.390 |
| caption_frames | 17.693 |
| sample_fps | 1.735 |
| detect_object_yolo | 5.940 |
| audio_scan | 15.623 |
| asr_timings | 8.535 |
| ast_timings | 13.032 |
| describe_scenes | 6.599 |
| summarize_scenes | 12.527 |
| synthesize_synopsis | 20.089 |
| make_embedding | 2.087 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 18.089 |
| branch_yolo_total | 7.681 |
| branch_audio_total | 37.198 |
