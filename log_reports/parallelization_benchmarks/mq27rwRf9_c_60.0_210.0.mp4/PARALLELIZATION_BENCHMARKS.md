# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 16:25:00 UTC | mq27rwRf9_c_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 182.306 | 0.635 | 66.670 | 14.916 | 11.387 | 7.691 | 5.435 |

## 2026-06-27 16:25:00 UTC | mq27rwRf9_c_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/mq27rwRf9_c_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `182.306` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.635 |
| save_clips | - |
| sample_frames | 1.680 |
| caption_frames | 58.516 |
| sample_fps | 2.439 |
| detect_object_yolo | 11.495 |
| audio_scan | 13.878 |
| asr_timings | 10.123 |
| ast_timings | 42.660 |
| describe_scenes | 14.916 |
| summarize_scenes | 11.387 |
| synthesize_synopsis | 7.691 |
| make_embedding | 5.435 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 60.203 |
| branch_yolo_total | 13.940 |
| branch_audio_total | 66.670 |
