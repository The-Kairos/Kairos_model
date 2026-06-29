# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 07:55:16 UTC | i9scCMPwu8I_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 124.382 | 0.697 | 43.340 | 13.381 | 10.434 | 18.464 | 2.543 |

## 2026-06-26 07:55:16 UTC | i9scCMPwu8I_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/i9scCMPwu8I_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `124.382` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.697 |
| save_clips | - |
| sample_frames | 0.670 |
| caption_frames | 23.896 |
| sample_fps | 1.897 |
| detect_object_yolo | 7.587 |
| audio_scan | 14.087 |
| asr_timings | 10.626 |
| ast_timings | 18.619 |
| describe_scenes | 13.381 |
| summarize_scenes | 10.434 |
| synthesize_synopsis | 18.464 |
| make_embedding | 2.543 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 24.573 |
| branch_yolo_total | 9.490 |
| branch_audio_total | 43.340 |
