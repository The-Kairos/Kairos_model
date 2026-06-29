# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 18:36:21 UTC | rUycc1YD41Q_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 163.385 | 0.798 | 53.646 | 17.650 | 18.634 | 14.063 | 3.623 |

## 2026-06-26 18:36:21 UTC | rUycc1YD41Q_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/rUycc1YD41Q_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `163.385` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.798 |
| save_clips | - |
| sample_frames | 1.245 |
| caption_frames | 40.763 |
| sample_fps | 2.316 |
| detect_object_yolo | 9.127 |
| audio_scan | 10.657 |
| asr_timings | 12.862 |
| ast_timings | 30.119 |
| describe_scenes | 17.650 |
| summarize_scenes | 18.634 |
| synthesize_synopsis | 14.063 |
| make_embedding | 3.623 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 42.015 |
| branch_yolo_total | 11.448 |
| branch_audio_total | 53.646 |
