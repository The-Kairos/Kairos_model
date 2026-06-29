# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 08:53:54 UTC | q_omrXXmkeE_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 151.026 | 0.799 | 52.750 | 12.906 | 17.012 | 9.658 | 3.554 |

## 2026-06-28 08:53:54 UTC | q_omrXXmkeE_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/q_omrXXmkeE_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `151.026` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.799 |
| save_clips | - |
| sample_frames | 1.078 |
| caption_frames | 40.169 |
| sample_fps | 2.310 |
| detect_object_yolo | 9.331 |
| audio_scan | 14.012 |
| asr_timings | 9.267 |
| ast_timings | 29.463 |
| describe_scenes | 12.906 |
| summarize_scenes | 17.012 |
| synthesize_synopsis | 9.658 |
| make_embedding | 3.554 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.253 |
| branch_yolo_total | 11.648 |
| branch_audio_total | 52.750 |
