# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 19:43:12 UTC | VFhXXjTwkKw_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 168.283 | 0.813 | 62.164 | 14.174 | 16.801 | 12.294 | 3.886 |

## 2026-06-25 19:43:12 UTC | VFhXXjTwkKw_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/VFhXXjTwkKw_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `168.283` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.813 |
| save_clips | - |
| sample_frames | 1.021 |
| caption_frames | 44.105 |
| sample_fps | 2.267 |
| detect_object_yolo | 9.379 |
| audio_scan | 15.975 |
| asr_timings | 11.747 |
| ast_timings | 34.433 |
| describe_scenes | 14.174 |
| summarize_scenes | 16.801 |
| synthesize_synopsis | 12.294 |
| make_embedding | 3.886 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.132 |
| branch_yolo_total | 11.651 |
| branch_audio_total | 62.164 |
