# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 03:29:05 UTC | eCvMNgspKxc_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 150.372 | 0.653 | 55.788 | 9.516 | 11.394 | 12.790 | 3.859 |

## 2026-06-26 03:29:05 UTC | eCvMNgspKxc_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/eCvMNgspKxc_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `150.372` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.653 |
| save_clips | - |
| sample_frames | 1.254 |
| caption_frames | 41.632 |
| sample_fps | 2.183 |
| detect_object_yolo | 9.887 |
| audio_scan | 10.855 |
| asr_timings | 11.416 |
| ast_timings | 33.509 |
| describe_scenes | 9.516 |
| summarize_scenes | 11.394 |
| synthesize_synopsis | 12.790 |
| make_embedding | 3.859 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 42.892 |
| branch_yolo_total | 12.075 |
| branch_audio_total | 55.788 |
