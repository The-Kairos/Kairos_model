# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 16:47:39 UTC | 8ETILC-7U1w_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 126.227 | 0.671 | 44.565 | 12.122 | 9.753 | 15.289 | 2.892 |

## 2026-06-24 16:47:39 UTC | 8ETILC-7U1w_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/8ETILC-7U1w_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `126.227` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.671 |
| save_clips | - |
| sample_frames | 1.006 |
| caption_frames | 29.016 |
| sample_fps | 2.051 |
| detect_object_yolo | 7.455 |
| audio_scan | 13.777 |
| asr_timings | 9.482 |
| ast_timings | 21.298 |
| describe_scenes | 12.122 |
| summarize_scenes | 9.753 |
| synthesize_synopsis | 15.289 |
| make_embedding | 2.892 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.028 |
| branch_yolo_total | 9.512 |
| branch_audio_total | 44.565 |
