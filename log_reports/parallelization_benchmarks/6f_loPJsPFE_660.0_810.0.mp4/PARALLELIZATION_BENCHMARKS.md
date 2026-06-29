# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 12:29:19 UTC | 6f_loPJsPFE_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 120.034 | 0.682 | 38.760 | 9.411 | 12.802 | 16.361 | 2.800 |

## 2026-06-24 12:29:19 UTC | 6f_loPJsPFE_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/6f_loPJsPFE_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `120.034` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.682 |
| save_clips | - |
| sample_frames | 0.721 |
| caption_frames | 27.096 |
| sample_fps | 1.941 |
| detect_object_yolo | 8.081 |
| audio_scan | 8.537 |
| asr_timings | 9.376 |
| ast_timings | 20.838 |
| describe_scenes | 9.411 |
| summarize_scenes | 12.802 |
| synthesize_synopsis | 16.361 |
| make_embedding | 2.800 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 27.822 |
| branch_yolo_total | 10.028 |
| branch_audio_total | 38.760 |
