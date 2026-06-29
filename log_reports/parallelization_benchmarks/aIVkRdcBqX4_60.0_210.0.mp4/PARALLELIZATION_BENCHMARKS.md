# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 00:22:15 UTC | aIVkRdcBqX4_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 89.595 | 0.732 | 35.413 | 4.862 | 6.237 | 15.052 | 1.821 |

## 2026-06-26 00:22:15 UTC | aIVkRdcBqX4_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/aIVkRdcBqX4_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `89.595` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.732 |
| save_clips | - |
| sample_frames | 0.299 |
| caption_frames | 16.112 |
| sample_fps | 1.785 |
| detect_object_yolo | 5.867 |
| audio_scan | 14.709 |
| asr_timings | 10.433 |
| ast_timings | 10.262 |
| describe_scenes | 4.862 |
| summarize_scenes | 6.237 |
| synthesize_synopsis | 15.052 |
| make_embedding | 1.821 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 16.417 |
| branch_yolo_total | 7.659 |
| branch_audio_total | 35.413 |
