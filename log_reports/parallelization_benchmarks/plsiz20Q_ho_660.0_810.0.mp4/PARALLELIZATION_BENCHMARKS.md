# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 08:19:15 UTC | plsiz20Q_ho_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 118.279 | 0.686 | 40.740 | 11.054 | 16.206 | 9.765 | 2.516 |

## 2026-06-28 08:19:15 UTC | plsiz20Q_ho_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/plsiz20Q_ho_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `118.279` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.686 |
| save_clips | - |
| sample_frames | 0.799 |
| caption_frames | 25.775 |
| sample_fps | 1.964 |
| detect_object_yolo | 7.368 |
| audio_scan | 12.797 |
| asr_timings | 9.163 |
| ast_timings | 18.772 |
| describe_scenes | 11.054 |
| summarize_scenes | 16.206 |
| synthesize_synopsis | 9.765 |
| make_embedding | 2.516 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 26.579 |
| branch_yolo_total | 9.338 |
| branch_audio_total | 40.740 |
