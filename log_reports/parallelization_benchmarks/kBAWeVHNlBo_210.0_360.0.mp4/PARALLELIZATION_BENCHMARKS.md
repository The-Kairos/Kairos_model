# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 13:17:10 UTC | kBAWeVHNlBo_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 263.326 | 0.787 | 64.418 | 22.170 | 78.038 | 27.471 | 4.484 |

## 2026-06-26 13:17:10 UTC | kBAWeVHNlBo_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/kBAWeVHNlBo_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `263.326` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.787 |
| save_clips | - |
| sample_frames | 1.181 |
| caption_frames | 50.432 |
| sample_fps | 2.388 |
| detect_object_yolo | 10.530 |
| audio_scan | 15.199 |
| asr_timings | 11.538 |
| ast_timings | 37.672 |
| describe_scenes | 22.170 |
| summarize_scenes | 78.038 |
| synthesize_synopsis | 27.471 |
| make_embedding | 4.484 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.619 |
| branch_yolo_total | 12.924 |
| branch_audio_total | 64.418 |
