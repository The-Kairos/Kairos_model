# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 01:57:45 UTC | v0x-YFvZXZY_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 166.080 | 0.812 | 61.634 | 13.204 | 10.140 | 7.192 | 4.712 |

## 2026-06-27 01:57:45 UTC | v0x-YFvZXZY_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/v0x-YFvZXZY_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `166.080` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.812 |
| save_clips | - |
| sample_frames | 1.516 |
| caption_frames | 52.589 |
| sample_fps | 2.575 |
| detect_object_yolo | 10.297 |
| audio_scan | 11.814 |
| asr_timings | 11.141 |
| ast_timings | 38.670 |
| describe_scenes | 13.204 |
| summarize_scenes | 10.140 |
| synthesize_synopsis | 7.192 |
| make_embedding | 4.712 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 54.110 |
| branch_yolo_total | 12.877 |
| branch_audio_total | 61.634 |
