# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 07:06:11 UTC | hp87nj0iTCQ_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 224.113 | 0.813 | 64.979 | 30.225 | 41.919 | 21.072 | 4.227 |

## 2026-06-26 07:06:11 UTC | hp87nj0iTCQ_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/hp87nj0iTCQ_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `224.113` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.813 |
| save_clips | - |
| sample_frames | 1.215 |
| caption_frames | 46.328 |
| sample_fps | 2.334 |
| detect_object_yolo | 9.546 |
| audio_scan | 13.966 |
| asr_timings | 15.480 |
| ast_timings | 35.524 |
| describe_scenes | 30.225 |
| summarize_scenes | 41.919 |
| synthesize_synopsis | 21.072 |
| make_embedding | 4.227 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.550 |
| branch_yolo_total | 11.886 |
| branch_audio_total | 64.979 |
