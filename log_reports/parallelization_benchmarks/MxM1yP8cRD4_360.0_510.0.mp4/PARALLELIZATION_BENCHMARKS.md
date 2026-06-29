# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 10:12:40 UTC | MxM1yP8cRD4_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 178.574 | 0.800 | 101.080 | 8.899 | 12.403 | 17.550 | 2.305 |

## 2026-06-25 10:12:40 UTC | MxM1yP8cRD4_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/MxM1yP8cRD4_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `178.574` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.800 |
| save_clips | - |
| sample_frames | 0.635 |
| caption_frames | 24.148 |
| sample_fps | 1.997 |
| detect_object_yolo | 7.343 |
| audio_scan | 15.254 |
| asr_timings | 69.803 |
| ast_timings | 16.014 |
| describe_scenes | 8.899 |
| summarize_scenes | 12.403 |
| synthesize_synopsis | 17.550 |
| make_embedding | 2.305 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 24.789 |
| branch_yolo_total | 9.346 |
| branch_audio_total | 101.080 |
