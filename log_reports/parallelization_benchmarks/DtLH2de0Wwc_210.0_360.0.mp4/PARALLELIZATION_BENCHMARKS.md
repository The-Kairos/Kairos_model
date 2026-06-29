# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 23:02:23 UTC | DtLH2de0Wwc_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 148.339 | 0.812 | 63.758 | 8.905 | 8.531 | 15.603 | 3.282 |

## 2026-06-24 23:02:23 UTC | DtLH2de0Wwc_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/DtLH2de0Wwc_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `148.339` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.812 |
| save_clips | - |
| sample_frames | 0.975 |
| caption_frames | 34.847 |
| sample_fps | 2.202 |
| detect_object_yolo | 8.011 |
| audio_scan | 13.999 |
| asr_timings | 22.388 |
| ast_timings | 27.362 |
| describe_scenes | 8.905 |
| summarize_scenes | 8.531 |
| synthesize_synopsis | 15.603 |
| make_embedding | 3.282 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 35.827 |
| branch_yolo_total | 10.219 |
| branch_audio_total | 63.758 |
