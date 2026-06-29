# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 23:27:47 UTC | E-nJHaHlyG8_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 172.005 | 0.898 | 64.905 | 11.705 | 8.028 | 9.575 | 5.064 |

## 2026-06-24 23:27:47 UTC | E-nJHaHlyG8_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/E-nJHaHlyG8_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `172.005` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.898 |
| save_clips | - |
| sample_frames | 1.611 |
| caption_frames | 55.314 |
| sample_fps | 2.636 |
| detect_object_yolo | 10.834 |
| audio_scan | 14.016 |
| asr_timings | 9.504 |
| ast_timings | 41.376 |
| describe_scenes | 11.705 |
| summarize_scenes | 8.028 |
| synthesize_synopsis | 9.575 |
| make_embedding | 5.064 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 56.931 |
| branch_yolo_total | 13.476 |
| branch_audio_total | 64.905 |
