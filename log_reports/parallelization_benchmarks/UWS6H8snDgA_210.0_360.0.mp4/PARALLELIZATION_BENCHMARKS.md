# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 19:02:12 UTC | UWS6H8snDgA_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 183.520 | 0.687 | 58.133 | 24.153 | 22.793 | 14.965 | 3.807 |

## 2026-06-25 19:02:12 UTC | UWS6H8snDgA_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/UWS6H8snDgA_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `183.520` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.687 |
| save_clips | - |
| sample_frames | 1.291 |
| caption_frames | 44.729 |
| sample_fps | 2.208 |
| detect_object_yolo | 9.348 |
| audio_scan | 13.882 |
| asr_timings | 11.125 |
| ast_timings | 33.118 |
| describe_scenes | 24.153 |
| summarize_scenes | 22.793 |
| synthesize_synopsis | 14.965 |
| make_embedding | 3.807 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.025 |
| branch_yolo_total | 11.562 |
| branch_audio_total | 58.133 |
