# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 13:00:41 UTC | 6t9xyE7kABg_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 150.058 | 0.816 | 47.565 | 12.294 | 15.898 | 20.578 | 3.358 |

## 2026-06-24 13:00:41 UTC | 6t9xyE7kABg_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/6t9xyE7kABg_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `150.058` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.816 |
| save_clips | - |
| sample_frames | 1.019 |
| caption_frames | 36.672 |
| sample_fps | 2.196 |
| detect_object_yolo | 8.273 |
| audio_scan | 12.776 |
| asr_timings | 7.923 |
| ast_timings | 26.857 |
| describe_scenes | 12.294 |
| summarize_scenes | 15.898 |
| synthesize_synopsis | 20.578 |
| make_embedding | 3.358 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.696 |
| branch_yolo_total | 10.475 |
| branch_audio_total | 47.565 |
