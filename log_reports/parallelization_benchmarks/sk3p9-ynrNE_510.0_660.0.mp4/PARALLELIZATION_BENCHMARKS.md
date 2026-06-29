# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 21:25:18 UTC | sk3p9-ynrNE_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 1880.822 | 0.804 | 1787.415 | 14.204 | 13.693 | 8.719 | 3.629 |

## 2026-06-26 21:25:18 UTC | sk3p9-ynrNE_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/sk3p9-ynrNE_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `1880.822` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.804 |
| save_clips | - |
| sample_frames | 1.173 |
| caption_frames | 38.892 |
| sample_fps | 2.247 |
| detect_object_yolo | 8.623 |
| audio_scan | 13.946 |
| asr_timings | 1743.137 |
| ast_timings | 30.324 |
| describe_scenes | 14.204 |
| summarize_scenes | 13.693 |
| synthesize_synopsis | 8.719 |
| make_embedding | 3.629 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 40.070 |
| branch_yolo_total | 10.875 |
| branch_audio_total | 1787.415 |
