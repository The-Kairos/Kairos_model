# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 09:46:14 UTC | iy6kh6tBCmI_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 127.143 | 0.781 | 42.614 | 12.592 | 15.376 | 17.442 | 2.635 |

## 2026-06-26 09:46:14 UTC | iy6kh6tBCmI_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/iy6kh6tBCmI_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `127.143` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.781 |
| save_clips | - |
| sample_frames | 0.566 |
| caption_frames | 24.198 |
| sample_fps | 1.975 |
| detect_object_yolo | 7.519 |
| audio_scan | 14.070 |
| asr_timings | 10.680 |
| ast_timings | 17.855 |
| describe_scenes | 12.592 |
| summarize_scenes | 15.376 |
| synthesize_synopsis | 17.442 |
| make_embedding | 2.635 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 24.769 |
| branch_yolo_total | 9.501 |
| branch_audio_total | 42.614 |
