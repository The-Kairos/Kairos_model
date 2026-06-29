# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 05:01:28 UTC | yQ5wwBumNG8_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 91.809 | 0.643 | 27.126 | 8.209 | 5.905 | 15.268 | 2.566 |

## 2026-06-27 05:01:28 UTC | yQ5wwBumNG8_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/yQ5wwBumNG8_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `91.809` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.643 |
| save_clips | - |
| sample_frames | 0.590 |
| caption_frames | 26.530 |
| sample_fps | 1.889 |
| detect_object_yolo | 6.817 |
| audio_scan | 3.875 |
| asr_timings | 0.000 |
| ast_timings | 18.105 |
| describe_scenes | 8.209 |
| summarize_scenes | 5.905 |
| synthesize_synopsis | 15.268 |
| make_embedding | 2.566 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 27.126 |
| branch_yolo_total | 8.712 |
| branch_audio_total | 21.988 |
