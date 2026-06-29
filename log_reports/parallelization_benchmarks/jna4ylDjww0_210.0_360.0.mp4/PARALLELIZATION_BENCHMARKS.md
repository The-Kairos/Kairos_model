# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 11:41:51 UTC | jna4ylDjww0_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 143.187 | 0.692 | 37.751 | 11.777 | 18.452 | 37.230 | 2.318 |

## 2026-06-26 11:41:51 UTC | jna4ylDjww0_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jna4ylDjww0_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `143.187` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.692 |
| save_clips | - |
| sample_frames | 0.548 |
| caption_frames | 23.306 |
| sample_fps | 1.901 |
| detect_object_yolo | 7.741 |
| audio_scan | 10.929 |
| asr_timings | 10.712 |
| ast_timings | 16.101 |
| describe_scenes | 11.777 |
| summarize_scenes | 18.452 |
| synthesize_synopsis | 37.230 |
| make_embedding | 2.318 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 23.859 |
| branch_yolo_total | 9.648 |
| branch_audio_total | 37.751 |
