# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 03:58:44 UTC | I96hjUvyk30_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 127.624 | 0.776 | 74.977 | 5.210 | 8.770 | 10.329 | 1.791 |

## 2026-06-25 03:58:44 UTC | I96hjUvyk30_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/I96hjUvyk30_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `127.624` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.776 |
| save_clips | - |
| sample_frames | 0.362 |
| caption_frames | 15.814 |
| sample_fps | 1.873 |
| detect_object_yolo | 6.326 |
| audio_scan | 12.781 |
| asr_timings | 52.057 |
| ast_timings | 10.129 |
| describe_scenes | 5.210 |
| summarize_scenes | 8.770 |
| synthesize_synopsis | 10.329 |
| make_embedding | 1.791 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 16.182 |
| branch_yolo_total | 8.205 |
| branch_audio_total | 74.977 |
