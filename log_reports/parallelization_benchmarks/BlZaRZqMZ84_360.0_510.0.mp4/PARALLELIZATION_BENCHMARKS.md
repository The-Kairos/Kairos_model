# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 19:46:28 UTC | BlZaRZqMZ84_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 116.468 | 0.715 | 37.171 | 9.206 | 20.668 | 15.393 | 2.483 |

## 2026-06-24 19:46:28 UTC | BlZaRZqMZ84_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/BlZaRZqMZ84_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `116.468` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.715 |
| save_clips | - |
| sample_frames | 0.502 |
| caption_frames | 15.895 |
| sample_fps | 1.808 |
| detect_object_yolo | 6.546 |
| audio_scan | 13.058 |
| asr_timings | 8.836 |
| ast_timings | 15.270 |
| describe_scenes | 9.206 |
| summarize_scenes | 20.668 |
| synthesize_synopsis | 15.393 |
| make_embedding | 2.483 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 16.402 |
| branch_yolo_total | 8.359 |
| branch_audio_total | 37.171 |
