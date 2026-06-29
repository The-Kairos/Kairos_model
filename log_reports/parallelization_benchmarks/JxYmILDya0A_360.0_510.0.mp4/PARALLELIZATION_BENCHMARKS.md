# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 05:28:14 UTC | JxYmILDya0A_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 166.943 | 0.798 | 55.451 | 17.125 | 11.491 | 15.434 | 4.141 |

## 2026-06-25 05:28:14 UTC | JxYmILDya0A_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/JxYmILDya0A_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `166.943` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.798 |
| save_clips | - |
| sample_frames | 1.572 |
| caption_frames | 47.285 |
| sample_fps | 2.421 |
| detect_object_yolo | 9.770 |
| audio_scan | 11.905 |
| asr_timings | 8.675 |
| ast_timings | 34.862 |
| describe_scenes | 17.125 |
| summarize_scenes | 11.491 |
| synthesize_synopsis | 15.434 |
| make_embedding | 4.141 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 48.863 |
| branch_yolo_total | 12.196 |
| branch_audio_total | 55.451 |
