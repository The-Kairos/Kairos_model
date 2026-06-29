# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 00:03:14 UTC | tvDH4JM_gME_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 191.176 | 0.803 | 62.162 | 17.998 | 19.792 | 8.248 | 5.345 |

## 2026-06-27 00:03:14 UTC | tvDH4JM_gME_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/tvDH4JM_gME_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `191.176` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.803 |
| save_clips | - |
| sample_frames | 1.627 |
| caption_frames | 60.226 |
| sample_fps | 2.580 |
| detect_object_yolo | 10.953 |
| audio_scan | 7.511 |
| asr_timings | 9.960 |
| ast_timings | 44.684 |
| describe_scenes | 17.998 |
| summarize_scenes | 19.792 |
| synthesize_synopsis | 8.248 |
| make_embedding | 5.345 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 61.859 |
| branch_yolo_total | 13.539 |
| branch_audio_total | 62.162 |
