# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 10:20:45 UTC | jAVGP5XwXns_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 177.242 | 0.657 | 50.887 | 18.160 | 30.494 | 26.979 | 3.097 |

## 2026-06-26 10:20:45 UTC | jAVGP5XwXns_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jAVGP5XwXns_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `177.242` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.657 |
| save_clips | - |
| sample_frames | 1.195 |
| caption_frames | 34.314 |
| sample_fps | 2.196 |
| detect_object_yolo | 7.849 |
| audio_scan | 15.011 |
| asr_timings | 11.314 |
| ast_timings | 24.543 |
| describe_scenes | 18.160 |
| summarize_scenes | 30.494 |
| synthesize_synopsis | 26.979 |
| make_embedding | 3.097 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 35.515 |
| branch_yolo_total | 10.050 |
| branch_audio_total | 50.887 |
