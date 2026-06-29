# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 16:21:57 UTC | mq27rwRf9_c_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 181.336 | 0.618 | 65.803 | 15.564 | 10.277 | 9.832 | 5.460 |

## 2026-06-27 16:21:57 UTC | mq27rwRf9_c_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/mq27rwRf9_c_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `181.336` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.618 |
| save_clips | - |
| sample_frames | 1.714 |
| caption_frames | 56.815 |
| sample_fps | 2.499 |
| detect_object_yolo | 11.381 |
| audio_scan | 13.730 |
| asr_timings | 8.877 |
| ast_timings | 43.188 |
| describe_scenes | 15.564 |
| summarize_scenes | 10.277 |
| synthesize_synopsis | 9.832 |
| make_embedding | 5.460 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 58.535 |
| branch_yolo_total | 13.886 |
| branch_audio_total | 65.803 |
