# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 06:53:26 UTC | hp87nj0iTCQ_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 132.937 | 0.790 | 46.081 | 9.609 | 17.054 | 15.065 | 2.819 |

## 2026-06-26 06:53:26 UTC | hp87nj0iTCQ_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/hp87nj0iTCQ_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `132.937` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.790 |
| save_clips | - |
| sample_frames | 0.688 |
| caption_frames | 29.696 |
| sample_fps | 2.060 |
| detect_object_yolo | 7.670 |
| audio_scan | 14.000 |
| asr_timings | 11.423 |
| ast_timings | 20.649 |
| describe_scenes | 9.609 |
| summarize_scenes | 17.054 |
| synthesize_synopsis | 15.065 |
| make_embedding | 2.819 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.390 |
| branch_yolo_total | 9.736 |
| branch_audio_total | 46.081 |
