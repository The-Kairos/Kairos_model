# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 14:58:42 UTC | l6aGqD9b53w_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 123.048 | 0.809 | 37.448 | 11.644 | 12.399 | 25.061 | 2.308 |

## 2026-06-26 14:58:42 UTC | l6aGqD9b53w_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/l6aGqD9b53w_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `123.048` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.809 |
| save_clips | - |
| sample_frames | 0.444 |
| caption_frames | 22.384 |
| sample_fps | 1.984 |
| detect_object_yolo | 7.142 |
| audio_scan | 12.975 |
| asr_timings | 8.261 |
| ast_timings | 16.203 |
| describe_scenes | 11.644 |
| summarize_scenes | 12.399 |
| synthesize_synopsis | 25.061 |
| make_embedding | 2.308 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 22.834 |
| branch_yolo_total | 9.132 |
| branch_audio_total | 37.448 |
