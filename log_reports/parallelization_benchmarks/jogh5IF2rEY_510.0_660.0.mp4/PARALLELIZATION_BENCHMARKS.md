# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 11:56:28 UTC | jogh5IF2rEY_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 151.861 | 0.808 | 41.790 | 17.458 | 29.246 | 21.681 | 2.558 |

## 2026-06-26 11:56:28 UTC | jogh5IF2rEY_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jogh5IF2rEY_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `151.861` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.808 |
| save_clips | - |
| sample_frames | 0.552 |
| caption_frames | 26.988 |
| sample_fps | 2.036 |
| detect_object_yolo | 7.331 |
| audio_scan | 10.782 |
| asr_timings | 12.226 |
| ast_timings | 18.774 |
| describe_scenes | 17.458 |
| summarize_scenes | 29.246 |
| synthesize_synopsis | 21.681 |
| make_embedding | 2.558 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 27.546 |
| branch_yolo_total | 9.372 |
| branch_audio_total | 41.790 |
