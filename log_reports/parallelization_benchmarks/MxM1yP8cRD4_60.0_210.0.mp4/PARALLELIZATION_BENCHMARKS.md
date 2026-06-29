# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 10:17:59 UTC | MxM1yP8cRD4_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 152.014 | 0.782 | 62.523 | 14.926 | 10.713 | 19.285 | 2.800 |

## 2026-06-25 10:17:59 UTC | MxM1yP8cRD4_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/MxM1yP8cRD4_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `152.014` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.782 |
| save_clips | - |
| sample_frames | 0.688 |
| caption_frames | 29.405 |
| sample_fps | 2.071 |
| detect_object_yolo | 7.377 |
| audio_scan | 11.768 |
| asr_timings | 29.394 |
| ast_timings | 21.352 |
| describe_scenes | 14.926 |
| summarize_scenes | 10.713 |
| synthesize_synopsis | 19.285 |
| make_embedding | 2.800 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.099 |
| branch_yolo_total | 9.454 |
| branch_audio_total | 62.523 |
