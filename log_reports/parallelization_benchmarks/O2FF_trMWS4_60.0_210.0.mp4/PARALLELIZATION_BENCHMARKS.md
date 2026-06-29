# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 11:24:30 UTC | O2FF_trMWS4_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 198.461 | 0.694 | 52.349 | 15.265 | 27.043 | 46.101 | 3.290 |

## 2026-06-25 11:24:30 UTC | O2FF_trMWS4_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/O2FF_trMWS4_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `198.461` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.694 |
| save_clips | - |
| sample_frames | 1.065 |
| caption_frames | 40.158 |
| sample_fps | 2.146 |
| detect_object_yolo | 8.868 |
| audio_scan | 15.291 |
| asr_timings | 8.880 |
| ast_timings | 28.169 |
| describe_scenes | 15.265 |
| summarize_scenes | 27.043 |
| synthesize_synopsis | 46.101 |
| make_embedding | 3.290 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.229 |
| branch_yolo_total | 11.020 |
| branch_audio_total | 52.349 |
