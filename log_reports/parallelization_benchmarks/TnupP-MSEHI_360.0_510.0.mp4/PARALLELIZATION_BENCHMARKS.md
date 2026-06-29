# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 18:03:19 UTC | TnupP-MSEHI_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 183.719 | 0.788 | 61.771 | 21.079 | 18.926 | 16.440 | 4.063 |

## 2026-06-25 18:03:19 UTC | TnupP-MSEHI_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/TnupP-MSEHI_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `183.719` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.788 |
| save_clips | - |
| sample_frames | 1.302 |
| caption_frames | 46.009 |
| sample_fps | 2.339 |
| detect_object_yolo | 9.596 |
| audio_scan | 15.004 |
| asr_timings | 11.070 |
| ast_timings | 35.689 |
| describe_scenes | 21.079 |
| summarize_scenes | 18.926 |
| synthesize_synopsis | 16.440 |
| make_embedding | 4.063 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.316 |
| branch_yolo_total | 11.942 |
| branch_audio_total | 61.771 |
