# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 13:52:46 UTC | 1icyCzbxmmg_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 195.012 | 0.663 | 50.175 | 30.800 | 23.020 | 37.772 | 3.368 |
| 2026-06-27 15:20:08 UTC | 1icyCzbxmmg_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 129.962 | 0.675 | 50.947 | 11.157 | 5.662 | 7.441 | 3.291 |

## 2026-06-23 13:52:46 UTC | 1icyCzbxmmg_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1icyCzbxmmg_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `195.012` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.663 |
| save_clips | - |
| sample_frames | 1.189 |
| caption_frames | 36.034 |
| sample_fps | 2.135 |
| detect_object_yolo | 8.445 |
| audio_scan | 14.802 |
| asr_timings | 8.306 |
| ast_timings | 27.059 |
| describe_scenes | 30.800 |
| summarize_scenes | 23.020 |
| synthesize_synopsis | 37.772 |
| make_embedding | 3.368 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.229 |
| branch_yolo_total | 10.586 |
| branch_audio_total | 50.175 |

## 2026-06-27 15:20:08 UTC | 1icyCzbxmmg_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1icyCzbxmmg_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `129.962` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.675 |
| save_clips | - |
| sample_frames | 1.208 |
| caption_frames | 37.571 |
| sample_fps | 2.157 |
| detect_object_yolo | 8.456 |
| audio_scan | 15.000 |
| asr_timings | 8.643 |
| ast_timings | 27.295 |
| describe_scenes | 11.157 |
| summarize_scenes | 5.662 |
| synthesize_synopsis | 7.441 |
| make_embedding | 3.291 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.785 |
| branch_yolo_total | 10.620 |
| branch_audio_total | 50.947 |
