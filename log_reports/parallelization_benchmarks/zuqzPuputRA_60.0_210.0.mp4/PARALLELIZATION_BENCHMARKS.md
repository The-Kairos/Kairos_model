# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 06:24:05 UTC | zuqzPuputRA_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 165.303 | 0.782 | 57.733 | 15.473 | 8.176 | 7.609 | 5.089 |

## 2026-06-27 06:24:05 UTC | zuqzPuputRA_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/zuqzPuputRA_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `165.303` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.782 |
| save_clips | - |
| sample_frames | 1.646 |
| caption_frames | 54.504 |
| sample_fps | 2.543 |
| detect_object_yolo | 10.347 |
| audio_scan | 8.588 |
| asr_timings | 7.848 |
| ast_timings | 41.289 |
| describe_scenes | 15.473 |
| summarize_scenes | 8.176 |
| synthesize_synopsis | 7.609 |
| make_embedding | 5.089 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 56.156 |
| branch_yolo_total | 12.895 |
| branch_audio_total | 57.733 |
