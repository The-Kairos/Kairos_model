# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 12:30:27 UTC | OkKEFnPQxPw_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 185.685 | 0.677 | 44.636 | 13.719 | 48.734 | 39.351 | 2.596 |

## 2026-06-25 12:30:27 UTC | OkKEFnPQxPw_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/OkKEFnPQxPw_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `185.685` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.677 |
| save_clips | - |
| sample_frames | 0.657 |
| caption_frames | 24.710 |
| sample_fps | 1.930 |
| detect_object_yolo | 7.222 |
| audio_scan | 16.620 |
| asr_timings | 9.613 |
| ast_timings | 18.393 |
| describe_scenes | 13.719 |
| summarize_scenes | 48.734 |
| synthesize_synopsis | 39.351 |
| make_embedding | 2.596 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 25.373 |
| branch_yolo_total | 9.158 |
| branch_audio_total | 44.636 |
