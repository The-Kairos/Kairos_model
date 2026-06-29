# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 06:19:43 UTC | zuqzPuputRA_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 110.779 | 0.818 | 40.409 | 10.052 | 6.030 | 5.386 | 3.034 |

## 2026-06-27 06:19:43 UTC | zuqzPuputRA_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/zuqzPuputRA_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `110.779` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.818 |
| save_clips | - |
| sample_frames | 1.110 |
| caption_frames | 32.591 |
| sample_fps | 2.305 |
| detect_object_yolo | 7.651 |
| audio_scan | 8.568 |
| asr_timings | 7.222 |
| ast_timings | 24.611 |
| describe_scenes | 10.052 |
| summarize_scenes | 6.030 |
| synthesize_synopsis | 5.386 |
| make_embedding | 3.034 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 33.708 |
| branch_yolo_total | 9.961 |
| branch_audio_total | 40.409 |
