# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 04:47:40 UTC | yMBJSWyKiDs_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 114.227 | 0.765 | 46.603 | 6.649 | 9.533 | 9.910 | 2.732 |

## 2026-06-27 04:47:40 UTC | yMBJSWyKiDs_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/yMBJSWyKiDs_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `114.227` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.765 |
| save_clips | - |
| sample_frames | 0.537 |
| caption_frames | 27.316 |
| sample_fps | 2.002 |
| detect_object_yolo | 6.783 |
| audio_scan | 11.884 |
| asr_timings | 15.927 |
| ast_timings | 18.783 |
| describe_scenes | 6.649 |
| summarize_scenes | 9.533 |
| synthesize_synopsis | 9.910 |
| make_embedding | 2.732 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 27.859 |
| branch_yolo_total | 8.790 |
| branch_audio_total | 46.603 |
