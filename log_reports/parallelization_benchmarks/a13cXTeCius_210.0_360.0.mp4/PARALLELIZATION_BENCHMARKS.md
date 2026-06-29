# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 00:07:22 UTC | a13cXTeCius_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 188.965 | 0.787 | 66.132 | 15.366 | 20.250 | 15.128 | 4.705 |

## 2026-06-26 00:07:22 UTC | a13cXTeCius_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/a13cXTeCius_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `188.965` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.787 |
| save_clips | - |
| sample_frames | 1.391 |
| caption_frames | 51.045 |
| sample_fps | 2.494 |
| detect_object_yolo | 10.256 |
| audio_scan | 15.028 |
| asr_timings | 11.945 |
| ast_timings | 39.151 |
| describe_scenes | 15.366 |
| summarize_scenes | 20.250 |
| synthesize_synopsis | 15.128 |
| make_embedding | 4.705 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 52.442 |
| branch_yolo_total | 12.756 |
| branch_audio_total | 66.132 |
