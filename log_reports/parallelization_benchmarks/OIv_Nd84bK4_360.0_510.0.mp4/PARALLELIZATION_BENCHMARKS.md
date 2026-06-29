# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 11:30:44 UTC | OIv_Nd84bK4_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 169.601 | 0.809 | 56.953 | 18.254 | 15.867 | 20.162 | 3.884 |

## 2026-06-25 11:30:44 UTC | OIv_Nd84bK4_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/OIv_Nd84bK4_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `169.601` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.809 |
| save_clips | - |
| sample_frames | 1.160 |
| caption_frames | 39.265 |
| sample_fps | 2.294 |
| detect_object_yolo | 9.547 |
| audio_scan | 12.595 |
| asr_timings | 10.382 |
| ast_timings | 33.969 |
| describe_scenes | 18.254 |
| summarize_scenes | 15.867 |
| synthesize_synopsis | 20.162 |
| make_embedding | 3.884 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 40.431 |
| branch_yolo_total | 11.846 |
| branch_audio_total | 56.953 |
