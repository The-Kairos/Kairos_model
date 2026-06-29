# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 23:10:20 UTC | DtLH2de0Wwc_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 148.498 | 0.825 | 65.563 | 8.496 | 6.561 | 9.756 | 3.562 |

## 2026-06-24 23:10:20 UTC | DtLH2de0Wwc_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/DtLH2de0Wwc_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `148.498` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.825 |
| save_clips | - |
| sample_frames | 1.115 |
| caption_frames | 40.074 |
| sample_fps | 2.252 |
| detect_object_yolo | 8.852 |
| audio_scan | 15.085 |
| asr_timings | 20.429 |
| ast_timings | 30.042 |
| describe_scenes | 8.496 |
| summarize_scenes | 6.561 |
| synthesize_synopsis | 9.756 |
| make_embedding | 3.562 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.196 |
| branch_yolo_total | 11.110 |
| branch_audio_total | 65.563 |
