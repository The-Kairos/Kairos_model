# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 22:09:50 UTC | 3UPP_WRL86c_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 141.278 | 0.769 | 81.578 | 5.562 | 8.233 | 8.012 | 2.540 |

## 2026-06-21 22:09:50 UTC | 3UPP_WRL86c_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3UPP_WRL86c_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `141.278` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.769 |
| save_clips | - |
| sample_frames | 0.693 |
| caption_frames | 23.743 |
| sample_fps | 2.009 |
| detect_object_yolo | 6.731 |
| audio_scan | 12.830 |
| asr_timings | 50.677 |
| ast_timings | 18.062 |
| describe_scenes | 5.562 |
| summarize_scenes | 8.233 |
| synthesize_synopsis | 8.012 |
| make_embedding | 2.540 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 24.442 |
| branch_yolo_total | 8.746 |
| branch_audio_total | 81.578 |
