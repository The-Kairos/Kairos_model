# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 19:37:14 UTC | BUxz53FWaj4_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 164.259 | 0.701 | 57.644 | 17.072 | 10.803 | 10.069 | 4.192 |

## 2026-06-24 19:37:14 UTC | BUxz53FWaj4_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/BUxz53FWaj4_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `164.259` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.701 |
| save_clips | - |
| sample_frames | 1.100 |
| caption_frames | 48.693 |
| sample_fps | 2.189 |
| detect_object_yolo | 10.330 |
| audio_scan | 12.923 |
| asr_timings | 9.536 |
| ast_timings | 35.177 |
| describe_scenes | 17.072 |
| summarize_scenes | 10.803 |
| synthesize_synopsis | 10.069 |
| make_embedding | 4.192 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.799 |
| branch_yolo_total | 12.525 |
| branch_audio_total | 57.644 |
