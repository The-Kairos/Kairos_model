# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 14:02:25 UTC | 7Q3Gpf51QFU_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 228.030 | 0.785 | 73.190 | 27.426 | 23.309 | 21.289 | 5.961 |

## 2026-06-24 14:02:25 UTC | 7Q3Gpf51QFU_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/7Q3Gpf51QFU_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `228.030` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.785 |
| save_clips | - |
| sample_frames | 1.604 |
| caption_frames | 58.834 |
| sample_fps | 2.633 |
| detect_object_yolo | 11.597 |
| audio_scan | 16.020 |
| asr_timings | 10.907 |
| ast_timings | 46.253 |
| describe_scenes | 27.426 |
| summarize_scenes | 23.309 |
| synthesize_synopsis | 21.289 |
| make_embedding | 5.961 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 60.444 |
| branch_yolo_total | 14.236 |
| branch_audio_total | 73.190 |
