# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 12:55:27 UTC | 6t9xyE7kABg_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 191.875 | 0.852 | 65.339 | 15.566 | 18.265 | 21.140 | 4.787 |

## 2026-06-24 12:55:27 UTC | 6t9xyE7kABg_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/6t9xyE7kABg_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `191.875` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.852 |
| save_clips | - |
| sample_frames | 1.567 |
| caption_frames | 50.306 |
| sample_fps | 2.472 |
| detect_object_yolo | 10.171 |
| audio_scan | 14.787 |
| asr_timings | 12.528 |
| ast_timings | 38.015 |
| describe_scenes | 15.566 |
| summarize_scenes | 18.265 |
| synthesize_synopsis | 21.140 |
| make_embedding | 4.787 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.879 |
| branch_yolo_total | 12.649 |
| branch_audio_total | 65.339 |
