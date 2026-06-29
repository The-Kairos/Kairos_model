# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 10:09:40 UTC | MxM1yP8cRD4_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 150.091 | 0.789 | 42.800 | 12.484 | 21.227 | 26.302 | 2.823 |

## 2026-06-25 10:09:40 UTC | MxM1yP8cRD4_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/MxM1yP8cRD4_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `150.091` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.789 |
| save_clips | - |
| sample_frames | 0.805 |
| caption_frames | 31.713 |
| sample_fps | 2.062 |
| detect_object_yolo | 7.665 |
| audio_scan | 11.946 |
| asr_timings | 7.545 |
| ast_timings | 23.301 |
| describe_scenes | 12.484 |
| summarize_scenes | 21.227 |
| synthesize_synopsis | 26.302 |
| make_embedding | 2.823 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.524 |
| branch_yolo_total | 9.733 |
| branch_audio_total | 42.800 |
