# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 11:57:21 UTC | 5Vc9wQIOkew_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 158.368 | 0.796 | 48.082 | 16.848 | 24.681 | 16.763 | 3.295 |

## 2026-06-24 11:57:21 UTC | 5Vc9wQIOkew_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/5Vc9wQIOkew_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `158.368` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.796 |
| save_clips | - |
| sample_frames | 1.156 |
| caption_frames | 35.137 |
| sample_fps | 2.287 |
| detect_object_yolo | 7.942 |
| audio_scan | 13.837 |
| asr_timings | 7.249 |
| ast_timings | 26.988 |
| describe_scenes | 16.848 |
| summarize_scenes | 24.681 |
| synthesize_synopsis | 16.763 |
| make_embedding | 3.295 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.298 |
| branch_yolo_total | 10.235 |
| branch_audio_total | 48.082 |
