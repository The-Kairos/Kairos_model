# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 08:13:01 UTC | plsiz20Q_ho_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 112.121 | 0.668 | 45.349 | 7.164 | 9.403 | 9.732 | 2.496 |

## 2026-06-28 08:13:01 UTC | plsiz20Q_ho_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/plsiz20Q_ho_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `112.121` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.668 |
| save_clips | - |
| sample_frames | 0.750 |
| caption_frames | 25.798 |
| sample_fps | 1.922 |
| detect_object_yolo | 7.398 |
| audio_scan | 15.053 |
| asr_timings | 12.001 |
| ast_timings | 18.279 |
| describe_scenes | 7.164 |
| summarize_scenes | 9.403 |
| synthesize_synopsis | 9.732 |
| make_embedding | 2.496 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 26.553 |
| branch_yolo_total | 9.325 |
| branch_audio_total | 45.349 |
