# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 21:20:28 UTC | WxcdmmJ5UzY_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 140.198 | 0.865 | 52.645 | 10.102 | 9.426 | 11.876 | 3.535 |

## 2026-06-25 21:20:28 UTC | WxcdmmJ5UzY_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/WxcdmmJ5UzY_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `140.198` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.865 |
| save_clips | - |
| sample_frames | 1.358 |
| caption_frames | 37.735 |
| sample_fps | 2.342 |
| detect_object_yolo | 8.896 |
| audio_scan | 13.974 |
| asr_timings | 8.774 |
| ast_timings | 29.888 |
| describe_scenes | 10.102 |
| summarize_scenes | 9.426 |
| synthesize_synopsis | 11.876 |
| make_embedding | 3.535 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.099 |
| branch_yolo_total | 11.244 |
| branch_audio_total | 52.645 |
