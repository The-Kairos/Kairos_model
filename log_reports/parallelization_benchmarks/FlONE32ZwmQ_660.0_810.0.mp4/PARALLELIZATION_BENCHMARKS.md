# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 00:31:06 UTC | FlONE32ZwmQ_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 121.418 | 0.643 | 47.094 | 8.245 | 6.338 | 14.153 | 2.745 |

## 2026-06-25 00:31:06 UTC | FlONE32ZwmQ_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/FlONE32ZwmQ_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `121.418` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.643 |
| save_clips | - |
| sample_frames | 0.753 |
| caption_frames | 30.145 |
| sample_fps | 1.957 |
| detect_object_yolo | 7.877 |
| audio_scan | 14.001 |
| asr_timings | 11.803 |
| ast_timings | 21.281 |
| describe_scenes | 8.245 |
| summarize_scenes | 6.338 |
| synthesize_synopsis | 14.153 |
| make_embedding | 2.745 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.903 |
| branch_yolo_total | 9.840 |
| branch_audio_total | 47.094 |
