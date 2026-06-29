# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 01:07:47 UTC | bUa-0ptWL5M_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 119.489 | 0.787 | 38.880 | 6.402 | 14.220 | 23.136 | 2.305 |

## 2026-06-26 01:07:47 UTC | bUa-0ptWL5M_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/bUa-0ptWL5M_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `119.489` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.787 |
| save_clips | - |
| sample_frames | 0.542 |
| caption_frames | 23.249 |
| sample_fps | 1.968 |
| detect_object_yolo | 6.572 |
| audio_scan | 13.770 |
| asr_timings | 9.154 |
| ast_timings | 15.947 |
| describe_scenes | 6.402 |
| summarize_scenes | 14.220 |
| synthesize_synopsis | 23.136 |
| make_embedding | 2.305 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 23.797 |
| branch_yolo_total | 8.546 |
| branch_audio_total | 38.880 |
