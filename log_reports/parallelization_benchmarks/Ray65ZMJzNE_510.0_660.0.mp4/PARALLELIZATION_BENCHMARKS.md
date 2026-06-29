# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 16:41:01 UTC | Ray65ZMJzNE_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 133.253 | 0.771 | 73.133 | 6.275 | 9.240 | 15.360 | 2.071 |

## 2026-06-25 16:41:01 UTC | Ray65ZMJzNE_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Ray65ZMJzNE_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `133.253` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.771 |
| save_clips | - |
| sample_frames | 0.429 |
| caption_frames | 16.224 |
| sample_fps | 1.905 |
| detect_object_yolo | 6.409 |
| audio_scan | 16.741 |
| asr_timings | 43.517 |
| ast_timings | 12.866 |
| describe_scenes | 6.275 |
| summarize_scenes | 9.240 |
| synthesize_synopsis | 15.360 |
| make_embedding | 2.071 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 16.658 |
| branch_yolo_total | 8.320 |
| branch_audio_total | 73.133 |
