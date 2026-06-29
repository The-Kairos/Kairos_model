# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 13:29:26 UTC | 0HAACVba7kI_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 116.828 | 0.836 | 44.359 | 7.741 | 11.569 | 7.741 | 2.864 |

## 2026-06-27 13:29:26 UTC | 0HAACVba7kI_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0HAACVba7kI_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `116.828` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.836 |
| save_clips | - |
| sample_frames | 1.037 |
| caption_frames | 29.421 |
| sample_fps | 2.160 |
| detect_object_yolo | 7.674 |
| audio_scan | 11.734 |
| asr_timings | 11.344 |
| ast_timings | 21.271 |
| describe_scenes | 7.741 |
| summarize_scenes | 11.569 |
| synthesize_synopsis | 7.741 |
| make_embedding | 2.864 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.464 |
| branch_yolo_total | 9.840 |
| branch_audio_total | 44.359 |
