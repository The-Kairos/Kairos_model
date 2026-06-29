# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 16:35:20 UTC | 7gVqrrCbcOw_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 204.288 | 0.801 | 63.262 | 28.842 | 25.961 | 17.223 | 4.497 |

## 2026-06-24 16:35:20 UTC | 7gVqrrCbcOw_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/7gVqrrCbcOw_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `204.288` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.801 |
| save_clips | - |
| sample_frames | 1.466 |
| caption_frames | 48.713 |
| sample_fps | 2.368 |
| detect_object_yolo | 9.777 |
| audio_scan | 15.979 |
| asr_timings | 10.197 |
| ast_timings | 37.077 |
| describe_scenes | 28.842 |
| summarize_scenes | 25.961 |
| synthesize_synopsis | 17.223 |
| make_embedding | 4.497 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 50.186 |
| branch_yolo_total | 12.151 |
| branch_audio_total | 63.262 |
