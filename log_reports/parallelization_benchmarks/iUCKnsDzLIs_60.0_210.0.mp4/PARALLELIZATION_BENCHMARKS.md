# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 09:01:36 UTC | iUCKnsDzLIs_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 228.223 | 0.813 | 69.439 | 23.322 | 23.325 | 28.379 | 6.025 |

## 2026-06-26 09:01:36 UTC | iUCKnsDzLIs_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/iUCKnsDzLIs_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `228.223` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.813 |
| save_clips | - |
| sample_frames | 1.657 |
| caption_frames | 59.967 |
| sample_fps | 2.694 |
| detect_object_yolo | 11.186 |
| audio_scan | 15.094 |
| asr_timings | 10.029 |
| ast_timings | 44.307 |
| describe_scenes | 23.322 |
| summarize_scenes | 23.325 |
| synthesize_synopsis | 28.379 |
| make_embedding | 6.025 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 61.630 |
| branch_yolo_total | 13.886 |
| branch_audio_total | 69.439 |
