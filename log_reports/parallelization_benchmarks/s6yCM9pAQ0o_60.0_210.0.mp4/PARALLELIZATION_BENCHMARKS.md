# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 19:22:00 UTC | s6yCM9pAQ0o_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 108.729 | 0.665 | 44.506 | 8.870 | 5.935 | 15.163 | 2.274 |

## 2026-06-26 19:22:00 UTC | s6yCM9pAQ0o_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/s6yCM9pAQ0o_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `108.729` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.665 |
| save_clips | - |
| sample_frames | 0.439 |
| caption_frames | 20.609 |
| sample_fps | 1.803 |
| detect_object_yolo | 7.030 |
| audio_scan | 15.856 |
| asr_timings | 12.969 |
| ast_timings | 15.672 |
| describe_scenes | 8.870 |
| summarize_scenes | 5.935 |
| synthesize_synopsis | 15.163 |
| make_embedding | 2.274 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 21.053 |
| branch_yolo_total | 8.839 |
| branch_audio_total | 44.506 |
