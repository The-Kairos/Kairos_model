# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 03:01:29 UTC | wQoQU0blZgw_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 151.423 | 0.787 | 56.489 | 9.802 | 10.200 | 9.019 | 4.105 |

## 2026-06-27 03:01:29 UTC | wQoQU0blZgw_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/wQoQU0blZgw_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `151.423` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.787 |
| save_clips | - |
| sample_frames | 1.292 |
| caption_frames | 46.176 |
| sample_fps | 2.366 |
| detect_object_yolo | 9.754 |
| audio_scan | 14.105 |
| asr_timings | 8.518 |
| ast_timings | 33.857 |
| describe_scenes | 9.802 |
| summarize_scenes | 10.200 |
| synthesize_synopsis | 9.019 |
| make_embedding | 4.105 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.474 |
| branch_yolo_total | 12.126 |
| branch_audio_total | 56.489 |
