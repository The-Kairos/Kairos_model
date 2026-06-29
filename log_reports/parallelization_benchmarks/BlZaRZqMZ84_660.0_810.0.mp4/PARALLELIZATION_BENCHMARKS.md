# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 19:52:34 UTC | BlZaRZqMZ84_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 138.909 | 0.680 | 52.874 | 11.245 | 8.397 | 13.200 | 3.280 |

## 2026-06-24 19:52:34 UTC | BlZaRZqMZ84_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/BlZaRZqMZ84_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `138.909` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.680 |
| save_clips | - |
| sample_frames | 0.924 |
| caption_frames | 35.802 |
| sample_fps | 1.984 |
| detect_object_yolo | 9.117 |
| audio_scan | 15.997 |
| asr_timings | 10.220 |
| ast_timings | 26.648 |
| describe_scenes | 11.245 |
| summarize_scenes | 8.397 |
| synthesize_synopsis | 13.200 |
| make_embedding | 3.280 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.732 |
| branch_yolo_total | 11.107 |
| branch_audio_total | 52.874 |
