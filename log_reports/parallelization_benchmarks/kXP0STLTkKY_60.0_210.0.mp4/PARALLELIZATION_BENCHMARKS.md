# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 13:55:33 UTC | kXP0STLTkKY_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 118.698 | 0.796 | 35.935 | 13.839 | 10.662 | 20.997 | 2.319 |

## 2026-06-26 13:55:33 UTC | kXP0STLTkKY_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/kXP0STLTkKY_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `118.698` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.796 |
| save_clips | - |
| sample_frames | 0.537 |
| caption_frames | 23.435 |
| sample_fps | 2.043 |
| detect_object_yolo | 6.716 |
| audio_scan | 8.672 |
| asr_timings | 11.044 |
| ast_timings | 16.210 |
| describe_scenes | 13.839 |
| summarize_scenes | 10.662 |
| synthesize_synopsis | 20.997 |
| make_embedding | 2.319 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 23.977 |
| branch_yolo_total | 8.765 |
| branch_audio_total | 35.935 |
